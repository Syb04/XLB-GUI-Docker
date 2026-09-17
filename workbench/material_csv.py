"""Strict CSV import for temperature-dependent material properties.

``parse_material_csv`` accepts two deliberately small CSV shapes.  A wide
file has one temperature column (``temperature_K`` or ``temperature_C``) and
one or more property columns.  A single-property file has a temperature
column and the literal ``value`` column; callers identify that column with
``property_key``.  Every non-blank record is validated, and a result is only
returned after the whole file has been accepted.

The canonical property names and their SI units are:

* ``density`` -- kg/m^3
* ``viscosity`` -- Pa*s (dynamic viscosity)
* ``heat_capacity`` -- J/(kg*K)
* ``conductivity`` -- W/(m*K)

The following exact property headers are also accepted and converted to the
canonical SI unit.  The aliases are intentionally explicit; arbitrary unit
expressions are not interpreted.

=====================  ================================================
Property               Accepted headers (in addition to the bare name)
=====================  ================================================
``density``            ``density_kg_m3``, ``density_kg_per_m3``,
                        ``density_g_cm3``, ``density_g_per_cm3``,
                        ``density_g_ml``, ``density_kg_l``,
                        ``density_kg_per_l``
``viscosity``          ``viscosity_Pa_s``, ``viscosity_kg_m_s``,
                        ``viscosity_mPa_s``, ``viscosity_cP``,
                        ``viscosity_pa_s``, ``viscosity_mpa_s``,
                        ``viscosity_cp``, ``dynamic_viscosity_Pa_s``,
                        ``dynamic_viscosity_pa_s``
``heat_capacity``      ``heat_capacity_J_kg_K``,
                        ``heat_capacity_J_kg_C``,
                        ``heat_capacity_kJ_kg_K``,
                        ``heat_capacity_kJ_kg_C``,
                        ``heat_capacity_j_kg_k``,
                        ``heat_capacity_j_kg_c``,
                        ``heat_capacity_kj_kg_k``,
                        ``heat_capacity_kj_kg_c``,
                        ``heat_capacity_J_per_kg_K``,
                        ``heat_capacity_kJ_per_kg_K``,
                        ``specific_heat_J_kg_K``,
                        ``specific_heat_kJ_kg_K``,
                        ``specific_heat_j_kg_k``,
                        ``specific_heat_kj_kg_k``
``conductivity``       ``conductivity_W_m_K``, ``conductivity_W_m_C``,
                        ``conductivity_mW_m_K``, ``conductivity_mW_m_C``,
                        ``conductivity_w_m_k``, ``conductivity_w_m_c``,
                        ``conductivity_mw_m_k``, ``conductivity_mw_m_c``,
                        ``conductivity_W_cm_K``, ``conductivity_W_per_m_K``,
                        ``conductivity_w_cm_k``,
                        ``conductivity_W_per_m_C``, ``conductivity_w_per_m_k``,
                        ``conductivity_w_per_m_c``,
                        ``thermal_conductivity_W_m_K``,
                        ``thermal_conductivity_w_m_k``
=====================  ================================================

Temperatures in Celsius are converted with ``T[K] = T[C] + 273.15``.  The
returned table always stores Kelvin temperatures, while ``temperature_unit``
records which temperature header was used.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any


MAX_ROWS = 10_000
MAX_TEXT_BYTES = 1 << 20
MIN_ROWS = 2

_TEMPERATURE_HEADERS = {"temperature_K": "K", "temperature_C": "C"}

# Header -> (canonical property name, multiplier to SI).  Keep this mapping
# explicit so that a typo such as ``density_kg_m2`` is rejected instead of
# being guessed at.
_PROPERTY_HEADERS: dict[str, tuple[str, float]] = {
    "density": ("density", 1.0),
    "density_kg_m3": ("density", 1.0),
    "density_kg_per_m3": ("density", 1.0),
    "density_g_cm3": ("density", 1_000.0),
    "density_g_per_cm3": ("density", 1_000.0),
    "density_g_ml": ("density", 1_000.0),
    "density_kg_l": ("density", 1_000.0),
    "density_kg_per_l": ("density", 1_000.0),
    "viscosity": ("viscosity", 1.0),
    "viscosity_Pa_s": ("viscosity", 1.0),
    "viscosity_kg_m_s": ("viscosity", 1.0),
    "viscosity_mPa_s": ("viscosity", 1.0e-3),
    "viscosity_cP": ("viscosity", 1.0e-3),
    "viscosity_pa_s": ("viscosity", 1.0),
    "viscosity_mpa_s": ("viscosity", 1.0e-3),
    "viscosity_cp": ("viscosity", 1.0e-3),
    "dynamic_viscosity_Pa_s": ("viscosity", 1.0),
    "dynamic_viscosity_pa_s": ("viscosity", 1.0),
    "heat_capacity": ("heat_capacity", 1.0),
    "heat_capacity_J_kg_K": ("heat_capacity", 1.0),
    "heat_capacity_J_kg_C": ("heat_capacity", 1.0),
    "heat_capacity_kJ_kg_K": ("heat_capacity", 1_000.0),
    "heat_capacity_kJ_kg_C": ("heat_capacity", 1_000.0),
    "heat_capacity_j_kg_k": ("heat_capacity", 1.0),
    "heat_capacity_j_kg_c": ("heat_capacity", 1.0),
    "heat_capacity_kj_kg_k": ("heat_capacity", 1_000.0),
    "heat_capacity_kj_kg_c": ("heat_capacity", 1_000.0),
    "heat_capacity_J_per_kg_K": ("heat_capacity", 1.0),
    "heat_capacity_kJ_per_kg_K": ("heat_capacity", 1_000.0),
    "specific_heat_J_kg_K": ("heat_capacity", 1.0),
    "specific_heat_kJ_kg_K": ("heat_capacity", 1_000.0),
    "specific_heat_j_kg_k": ("heat_capacity", 1.0),
    "specific_heat_kj_kg_k": ("heat_capacity", 1_000.0),
    "conductivity": ("conductivity", 1.0),
    "conductivity_W_m_K": ("conductivity", 1.0),
    "conductivity_W_m_C": ("conductivity", 1.0),
    "conductivity_mW_m_K": ("conductivity", 1.0e-3),
    "conductivity_mW_m_C": ("conductivity", 1.0e-3),
    "conductivity_w_m_k": ("conductivity", 1.0),
    "conductivity_w_m_c": ("conductivity", 1.0),
    "conductivity_mw_m_k": ("conductivity", 1.0e-3),
    "conductivity_mw_m_c": ("conductivity", 1.0e-3),
    "conductivity_W_cm_K": ("conductivity", 100.0),
    "conductivity_w_cm_k": ("conductivity", 100.0),
    "conductivity_W_per_m_K": ("conductivity", 1.0),
    "conductivity_W_per_m_C": ("conductivity", 1.0),
    "conductivity_w_per_m_k": ("conductivity", 1.0),
    "conductivity_w_per_m_c": ("conductivity", 1.0),
    "thermal_conductivity_W_m_K": ("conductivity", 1.0),
    "thermal_conductivity_w_m_k": ("conductivity", 1.0),
}

_PROPERTY_KEYS = frozenset({"density", "viscosity", "heat_capacity", "conductivity"})


def _error(message: str) -> ValueError:
    """Build the one exception type used for input validation failures."""

    return ValueError(message)


def _decode_text(text: str | bytes) -> str:
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _error("CSV text must be valid UTF-8") from exc
    if not isinstance(text, str):
        raise TypeError("text must be a str or UTF-8 bytes object")
    return text


def _number(raw: str, label: str, *, positive: bool = True) -> float:
    value_text = raw.strip()
    if not value_text:
        raise _error(f"{label} is missing")
    try:
        value = float(value_text)
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} must be a finite number") from exc
    if not math.isfinite(value):
        raise _error(f"{label} must be a finite number")
    if positive and value <= 0.0:
        raise _error(f"{label} must be positive")
    return value


def _header_name(raw: Any) -> str:
    # Trimming surrounding whitespace handles the common CSV form produced by
    # spreadsheet programs while keeping the unit spelling itself exact.
    return str(raw).strip()


def _normalise_property_key(property_key: Any) -> str:
    if not isinstance(property_key, str) or not property_key.strip():
        raise _error("property_key is required for a single-property CSV")
    key = property_key.strip()
    if key not in _PROPERTY_KEYS:
        names = ", ".join(sorted(_PROPERTY_KEYS))
        raise _error(f"unknown property_key {key!r}; expected one of {names}")
    return key


def _read_records(text: str) -> list[tuple[int, list[str]]]:
    """Read all CSV records, retaining line numbers for useful errors."""

    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, strict=True)
    records: list[tuple[int, list[str]]] = []
    try:
        for row in reader:
            if not row:
                continue
            records.append((reader.line_num, row))
    except csv.Error as exc:
        raise _error(f"invalid CSV near line {reader.line_num}: {exc}") from exc
    return records


def parse_material_csv(text: str | bytes, property_key: str | None = None) -> dict[str, Any]:
    """Parse a strict temperature/property CSV into SI property tables.

    Parameters
    ----------
    text:
        UTF-8 CSV text (a UTF-8 ``bytes`` object is accepted as a convenience).
    property_key:
        Canonical property name (``density``, ``viscosity``,
        ``heat_capacity``, or ``conductivity``), required only for the
        two-column ``temperature_*,value`` form.

    Raises
    ------
    ValueError
        If headers, records, units, temperatures, or values are invalid.
    TypeError
        If ``text`` is not text/UTF-8 bytes.
    """

    source = _decode_text(text)
    if len(source.encode("utf-8")) > MAX_TEXT_BYTES:
        raise _error(f"CSV text exceeds the {MAX_TEXT_BYTES}-byte limit")

    # A BOM is part of decoded UTF-8 text when callers pass a string rather
    # than bytes.  It is valid only at the beginning of the first header cell.
    if source.startswith("\ufeff"):
        source = source[1:]

    records = _read_records(source)
    if not records:
        raise _error("CSV must contain an explicit header row")

    header_line, raw_headers = records[0]
    if len(raw_headers) > len(_PROPERTY_KEYS) + 1:
        raise _error(
            "header contains too many columns; at most four properties and one temperature column are supported"
        )
    headers = [_header_name(value) for value in raw_headers]
    if not headers or any(not value for value in headers):
        raise _error(f"header row at line {header_line} contains a missing column name")
    if len(set(headers)) != len(headers):
        duplicates = sorted({name for name in headers if headers.count(name) > 1})
        raise _error(f"duplicate column(s) in header: {', '.join(duplicates)}")

    temperature_columns = [
        (index, name, _TEMPERATURE_HEADERS[name])
        for index, name in enumerate(headers)
        if name in _TEMPERATURE_HEADERS
    ]
    if not temperature_columns:
        raise _error("header must contain exactly one explicit temperature_K or temperature_C column")
    if len(temperature_columns) > 1:
        raise _error("conflicting temperature headers; use exactly one of temperature_K or temperature_C")
    temperature_index, _, temperature_unit = temperature_columns[0]

    # Resolve the columns before touching any data row.  This makes all schema
    # errors deterministic and ensures no partial result can escape.
    property_columns: list[tuple[int, str, str, float]] = []
    value_columns = [index for index, name in enumerate(headers) if name == "value"]
    is_single = len(headers) == 2 and len(value_columns) == 1
    if is_single:
        if property_key is None:
            raise _error("property_key is required for a single-property CSV")
        canonical = _normalise_property_key(property_key)
        value_index = value_columns[0]
        if value_index == temperature_index:
            raise _error("single-property CSV must contain temperature and value columns")
        property_columns.append((value_index, canonical, "value", 1.0))
    else:
        if property_key is not None:
            # It is useful to catch a typo early even when a caller supplies a
            # key defensively for a wide file; the key does not filter columns.
            _normalise_property_key(property_key)
        for index, name in enumerate(headers):
            if index == temperature_index:
                continue
            if name not in _PROPERTY_HEADERS:
                raise _error(f"unknown column {name!r}")
            canonical, multiplier = _PROPERTY_HEADERS[name]
            if any(item[1] == canonical for item in property_columns):
                raise _error(f"duplicate property column for {canonical!r}")
            property_columns.append((index, canonical, name, multiplier))
        if not property_columns:
            raise _error("header must contain at least one recognized property column")

    # A two-column form is single-property only when the non-temperature
    # column is exactly ``value``.  Conversely, a value column in a wider file
    # is an unknown column and should never be silently interpreted.
    if not is_single and value_columns:
        raise _error("unknown column 'value'; use property headers in a wide CSV")

    data_records = records[1:]
    if not data_records:
        raise _error("CSV must contain at least two data rows")
    if len(data_records) > MAX_ROWS:
        raise _error(f"CSV contains more than {MAX_ROWS} data rows")

    # Build local tables only after the complete header has passed validation.
    # No caller-owned object is ever mutated (the input itself is immutable).
    rows: list[tuple[float, list[float], int]] = []
    previous_temperature: float | None = None
    unordered = False
    seen_temperatures: set[float] = set()
    expected_columns = len(headers)
    for line_number, row in data_records:
        if len(row) != expected_columns:
            raise _error(
                f"line {line_number} has {len(row)} columns; expected {expected_columns}"
            )
        raw_temperature = row[temperature_index]
        # Celsius values may be negative; both scales are checked explicitly
        # below so Kelvin failures identify the physical constraint clearly.
        temperature = _number(raw_temperature, f"line {line_number} temperature", positive=False)
        if temperature_unit == "C":
            temperature += 273.15
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise _error(f"line {line_number} temperature is below absolute zero")
        elif temperature <= 0.0:
            raise _error(f"line {line_number} Kelvin temperature must be positive")
        if temperature in seen_temperatures:
            raise _error(f"line {line_number} duplicates a temperature")
        seen_temperatures.add(temperature)
        if previous_temperature is not None and temperature < previous_temperature:
            unordered = True
        previous_temperature = temperature

        values: list[float] = []
        for index, canonical, column_name, multiplier in property_columns:
            value = _number(row[index], f"line {line_number} column {column_name!r}")
            value *= multiplier
            if not math.isfinite(value) or value <= 0.0:
                raise _error(f"line {line_number} column {column_name!r} must be positive and finite")
            values.append(value)
        rows.append((temperature, values, line_number))

    if len(rows) < MIN_ROWS:
        raise _error("CSV must contain at least two data rows")

    if unordered:
        rows.sort(key=lambda item: item[0])
        warning = "Rows were unordered; points were sorted by temperature."
        warnings = [warning]
    else:
        warnings = []

    properties: dict[str, dict[str, Any]] = {}
    for property_position, (_, canonical, _, _) in enumerate(property_columns):
        properties[canonical] = {
            "kind": "table",
            "points": [
                [temperature, values[property_position]]
                for temperature, values, _ in rows
            ],
        }

    return {
        "properties": properties,
        "rows": len(rows),
        "temperature_unit": temperature_unit,
        "warnings": warnings,
    }


__all__ = ["MAX_ROWS", "MAX_TEXT_BYTES", "MIN_ROWS", "parse_material_csv"]
