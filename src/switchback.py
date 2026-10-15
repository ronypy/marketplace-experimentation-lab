"""Cell building, interference simulation, and switchback A/A helpers for Part B."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_uber_manhattan_trips(parquet_path, zone_lookup_path, n_zones=20):
    """Load one month of HVFHV data, keep Uber trips with a valid wait time in the
    top-N busiest Manhattan zones. Returns the trip-level dataframe (one row per trip)."""
    cols = ["hvfhs_license_num", "request_datetime", "pickup_datetime", "PULocationID"]
    t = pd.read_parquet(
        parquet_path, columns=cols,
        filters=[("hvfhs_license_num", "==", "HV0003")],
    )
    t["wait_min"] = (t.pickup_datetime - t.request_datetime).dt.total_seconds() / 60
    t = t[(t.wait_min > 0) & (t.wait_min < 60)]

    zones = pd.read_csv(zone_lookup_path)
    manh = zones[zones.Borough == "Manhattan"].LocationID
    t = t[t.PULocationID.isin(manh)]

    top = t.PULocationID.value_counts().head(n_zones).index
    t = t[t.PULocationID.isin(top)].copy()
    return t


def build_cells(trips):
    """Aggregate trip-level data into (zone, hour) cells with demand D and mean wait."""
    trips = trips.copy()
    trips["hour"] = trips.request_datetime.dt.floor("h")
    cells = (
        trips.groupby(["PULocationID", "hour"])
        .agg(D=("wait_min", "size"), wait=("wait_min", "mean"))
        .reset_index()
    )
    cells["how"] = cells.hour.dt.dayofweek * 24 + cells.hour.dt.hour
    return cells


def add_supply_cap(cells, quantile=0.8):
    """80th (or other) percentile of D by (zone, hour-of-week) across the month, as a supply cap."""
    cells = cells.copy()
    cells["S"] = cells.groupby(["PULocationID", "how"]).D.transform(lambda s: s.quantile(quantile))
    return cells


def ground_truth_effect(cells, L=0.10):
    """Global treatment effect: everyone treated vs nobody treated, under the supply cap."""
    return np.minimum((1 + L) * cells.D, cells.S).sum() / np.minimum(cells.D, cells.S).sum() - 1


def rider_ab_once(cells, L, rng):
    """One draw of a rider-level 50/50 A/B test in the same market (shared supply).
    Returns the estimated relative lift, plus the realized treated/control completions
    and demand, so callers can also measure harm to control riders."""
    dem_t = rng.poisson(0.5 * (1 + L) * cells.D)
    dem_c = rng.poisson(0.5 * cells.D)
    p = np.minimum(1, cells.S / np.maximum(dem_t + dem_c, 1))
    comp_t, comp_c = (dem_t * p).sum(), (dem_c * p).sum()
    estimate = comp_t / comp_c - 1
    return estimate, comp_t, comp_c, dem_t.sum(), dem_c.sum()


def switchback_once(cells, L, rng, block_h=2):
    """One draw of a switchback: randomize whole (zone, time-block) cells to treated/control."""
    blk = ((cells.hour - cells.hour.min()).dt.total_seconds() // (3600 * block_h)).astype(int)
    key = cells.PULocationID.astype(str) + "_" + blk.astype(str)
    assign = {k: rng.integers(0, 2) for k in key.unique()}
    T = key.map(assign).values
    dem = rng.poisson((1 + L * T) * cells.D)
    comp = np.minimum(dem, cells.S)
    d = cells.assign(T=T, comp=comp)
    fit = smf.ols("comp ~ T + C(PULocationID) + C(how)", data=d).fit()
    return fit.params["T"] / d.loc[d["T"] == 0, "comp"].mean()


def aa_block_length_once(cells, rng, block_h, trip_wait_var):
    """One draw of a switchback A/A test on the real `wait` series (no injected effect).
    Returns (estimate_minutes, naive_se_minutes, n_t_trips, n_c_trips)."""
    blk = ((cells.hour - cells.hour.min()).dt.total_seconds() // (3600 * block_h)).astype(int)
    key = cells.PULocationID.astype(str) + "_" + blk.astype(str)
    assign = {k: rng.integers(0, 2) for k in key.unique()}
    T = key.map(assign).values
    d = cells.assign(T=T)

    fit = smf.ols("wait ~ T + C(PULocationID) + C(how)", data=d).fit()
    estimate = fit.params["T"]

    n_t = d.loc[d["T"] == 1, "D"].sum()
    n_c = d.loc[d["T"] == 0, "D"].sum()
    naive_se = np.sqrt(trip_wait_var * (1 / n_t + 1 / n_c))
    return estimate, naive_se, n_t, n_c
