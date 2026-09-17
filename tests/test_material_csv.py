import unittest

from workbench.material_csv import MAX_ROWS, MAX_TEXT_BYTES, parse_material_csv


class MaterialCsvTests(unittest.TestCase):
    def test_wide_csv_converts_units_and_sorts_rows(self):
        csv_text = (
            "temperature_C,density_g_cm3,viscosity_mPa_s,"
            "heat_capacity_kJ_kg_K,conductivity_W_m_K\r\n"
            "40,0.992,0.653,4.18,0.63\r\n"
            "20,0.998,1.002,4.19,0.60\r\n"
        )

        result = parse_material_csv(csv_text)

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["temperature_unit"], "C")
        self.assertEqual(
            list(result["properties"]),
            ["density", "viscosity", "heat_capacity", "conductivity"],
        )
        self.assertEqual(result["properties"]["density"]["kind"], "table")
        self.assertAlmostEqual(result["properties"]["density"]["points"][0][0], 293.15)
        self.assertAlmostEqual(result["properties"]["density"]["points"][0][1], 998.0)
        self.assertAlmostEqual(result["properties"]["viscosity"]["points"][0][1], 1.002e-3)
        self.assertAlmostEqual(result["properties"]["heat_capacity"]["points"][1][1], 4180.0)
        self.assertAlmostEqual(result["properties"]["conductivity"]["points"][1][1], 0.63)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("sorted", result["warnings"][0])

    def test_single_property_csv_supports_bom_quotes_and_scientific_notation(self):
        csv_text = "\ufefftemperature_K,value\r\n2.80e2,\"8.00e-4\"\r\n3.20e2,\"1.20e-3\"\r\n"

        result = parse_material_csv(csv_text, property_key="viscosity")

        self.assertEqual(result["temperature_unit"], "K")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(
            result["properties"]["viscosity"]["points"],
            [[280.0, 8.00e-4], [320.0, 1.20e-3]],
        )

    def test_single_property_requires_a_canonical_property_key(self):
        with self.assertRaisesRegex(ValueError, "unknown property_key"):
            parse_material_csv(
                "temperature_K,value\n280,0.8\n320,1.2\n",
                property_key="viscosity_mPa_s",
            )

    def test_blank_lines_are_ignored_but_malformed_rows_are_rejected(self):
        result = parse_material_csv("\ntemperature_K,density\n\n280,900\n320,1000\n")
        self.assertEqual(result["rows"], 2)

        with self.assertRaisesRegex(ValueError, "columns"):
            parse_material_csv("temperature_K,density\n280,900\n320\n")

    def test_headers_must_be_explicit_and_known(self):
        cases = [
            ("280,900\n320,1000\n", None, "temperature"),
            ("temperature_K,other\n280,900\n320,1000\n", None, "unknown column"),
            ("temperature_K,temperature_C,density\n280,20,900\n320,40,1000\n", None, "conflicting"),
            ("temperature_K,density,density_kg_m3\n280,900,900\n320,1000,1000\n", None, "duplicate"),
        ]
        for csv_text, property_key, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_material_csv(csv_text, property_key)

    def test_single_property_requires_a_valid_property_key(self):
        csv_text = "temperature_K,value\n280,900\n320,1000\n"
        with self.assertRaisesRegex(ValueError, "property_key"):
            parse_material_csv(csv_text)
        with self.assertRaisesRegex(ValueError, "unknown property_key"):
            parse_material_csv(csv_text, "temperature")

    def test_invalid_values_temperatures_duplicates_and_short_files(self):
        invalid = [
            ("temperature_K,density\n0,900\n320,1000\n", "Kelvin"),
            ("temperature_C,density\n-273.15,900\n20,1000\n", "absolute zero"),
            ("temperature_K,density\n280,900\n280,1000\n", "duplicates"),
            ("temperature_K,density\n280,0\n320,1000\n", "positive"),
            ("temperature_K,density\n280,NaN\n320,1000\n", "finite"),
            ("temperature_K,density\n280,900\n", "two data rows"),
        ]
        for csv_text, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_material_csv(csv_text)

    def test_limits_are_enforced(self):
        many_rows = "temperature_K,density\n" + "".join(
            f"{280 + index / 100},{900 + index}\n" for index in range(MAX_ROWS + 1)
        )
        with self.assertRaisesRegex(ValueError, "more than"):
            parse_material_csv(many_rows)

        with self.assertRaisesRegex(ValueError, "1048576"):
            parse_material_csv("temperature_K,density\n" + "a" * MAX_TEXT_BYTES)

    def test_bytes_input_and_result_are_independent(self):
        result = parse_material_csv(b"temperature_K,density\n280,900\n320,1000\n")
        result["properties"]["density"]["points"][0][1] = 1.0
        again = parse_material_csv("temperature_K,density\n280,900\n320,1000\n")
        self.assertEqual(again["properties"]["density"]["points"][0][1], 900.0)


if __name__ == "__main__":
    unittest.main()
