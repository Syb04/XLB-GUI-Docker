"""Regenerate bundled SI tables with CoolProp 7.2.0 (development only).

PYTHONPATH=/tmp/xlb-property-tools-20260917 python scripts/generate_material_presets.py
Runtime installations do not need CoolProp; the generated JSON is bundled.
"""
import csv
import json
from pathlib import Path

import CoolProp
from CoolProp.CoolProp import PropsSI, PhaseSI

ROOT = Path(__file__).resolve().parents[1]
PRESSURE = 101325.0
PROPERTIES = {'density': 'Dmass', 'viscosity': 'VISCOSITY',
              'heat_capacity': 'Cpmass', 'conductivity': 'CONDUCTIVITY'}


def main():
    if CoolProp.__version__ != '7.2.0':
        raise RuntimeError('Reproducible generation requires CoolProp==7.2.0')
    presets = []
    for fluid, label, temperatures, phase in [
        ('Water', '水', range(5, 96, 5), '液体・5～95 °C'),
        ('Air', '乾燥空気', range(-50, 301, 10), '気体・−50～300 °C'),
    ]:
        rows = []
        for celsius in temperatures:
            kelvin = round(celsius + 273.15, 8)
            fluid_phase = PhaseSI('T', kelvin, 'P', PRESSURE, fluid)
            assert fluid_phase == 'liquid' if fluid == 'Water' else fluid_phase in ('gas', 'supercritical_gas')
            rows.append([kelvin] + [float(f'{PropsSI(output, "T", kelvin, "P", PRESSURE, fluid):.10g}')
                                    for output in PROPERTIES.values()])
        sources = [{'title': f'CoolProp {fluid} — 使用式・文献',
                    'url': f'https://coolprop.org/fluid_properties/fluids/{fluid}.html'}]
        common = {'pressure_pa': PRESSURE, 'sources': sources,
                  'generator': {'library': 'CoolProp', 'version': CoolProp.__version__,
                                'git_revision': CoolProp.__gitrevision__, 'backend': 'HEOS'}}
        key = fluid.lower()
        table = {'name': f'{label}（温度依存・1 atm）'}
        for column, property_key in enumerate(PROPERTIES, 1):
            table[property_key] = {'kind': 'table', 'points': [[row[0], row[column]] for row in rows]}
        presets.append({**common, 'id': key + '-temperature', 'label': f'{label}：温度依存',
                        'description': f'{phase}。101.325 kPa固定で生成した密度・粘度・定圧比熱・熱伝導率。温度間は線形補間、範囲外は端点値。',
                        'temperature_range_k': [rows[0][0], rows[-1][0]], 'material': table})
        constant = {'name': f'{label}（20 °C・定数）'}
        at_twenty = next(row for row in rows if row[0] == 293.15)
        for column, property_key in enumerate(PROPERTIES, 1):
            constant[property_key] = {'kind': 'constant', 'value': at_twenty[column]}
        presets.append({**common, 'id': key + '-constant', 'label': f'{label}：20 °C 定数',
                        'description': '20 °C・101.325 kPaの値を全温度で固定して使用します。',
                        'temperature_range_k': None, 'reference_temperature_k': 293.15, 'material': constant})
        example_dir = ROOT / 'static' / 'examples'
        example_dir.mkdir(parents=True, exist_ok=True)
        with (example_dir / f'{key}-temperature.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['temperature_K', *PROPERTIES])
            writer.writerows(rows)
    destination = ROOT / 'workbench' / 'data' / 'material_presets.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'version': 1, 'presets': presets}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Short valid templates, generated from the same traceable water dataset.
    water_rows = list(csv.reader((ROOT / 'static/examples/water-temperature.csv').open(encoding='utf-8')))
    with (ROOT / 'static/examples/material-properties.csv').open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerows([water_rows[0], *water_rows[1:4]])
    with (ROOT / 'static/examples/material-property.csv').open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerows([['temperature_K', 'value'], *[[row[0], row[2]] for row in water_rows[1:4]]])
    print(f'Wrote {len(presets)} presets to {destination}')


if __name__ == '__main__':
    main()
