"""Tests for Korean name normalisation and matching keys.

The cases here are all real players whose names appear differently across
sources. They are the regression suite for the hardest part of the crosswalk.
"""

import unittest

from kbo_mlb import names


class TestHangul(unittest.TestCase):
    def test_detects_hangul(self):
        self.assertTrue(names.has_hangul("이정후"))
        self.assertFalse(names.has_hangul("Lee Jung-hoo"))

    def test_romanization(self):
        self.assertEqual(names.romanize_hangul("이정후"), "i jeong hu")
        self.assertEqual(names.romanize_hangul("류현진"), "ryu hyeon jin")
        self.assertEqual(names.romanize_hangul("김하성"), "gim ha seong")

    def test_non_hangul_passes_through(self):
        self.assertEqual(names.romanize_hangul("Fedde"), "Fedde")


class TestCleanName(unittest.TestCase):
    def test_strips_handedness_markers(self):
        # '*' is left-handed, '#' switch-hitter. Neither is part of the name.
        self.assertEqual(names.clean_name("Jung-hoo Lee*"), "Jung-hoo Lee")
        self.assertEqual(names.clean_name("Ha-seong Kim#"), "Ha-seong Kim")

    def test_keeps_accents(self):
        self.assertEqual(names.clean_name("Sócrates Brito*"), "Sócrates Brito")


class TestSurnameSplitting(unittest.TestCase):
    def test_korean_order(self):
        self.assertEqual(names.split_surname("Lee Jung-hoo")[0], "LEE")

    def test_western_order(self):
        self.assertEqual(names.split_surname("Jung-hoo Lee")[0], "LEE")

    def test_given_name_syllable_is_not_mistaken_for_surname(self):
        # 'Jung' is a real surname, but here it is the first half of the
        # given name. Getting this wrong silently corrupts the whole join.
        self.assertEqual(names.split_surname("Jung Ho Kang")[0], "KANG")
        self.assertEqual(names.split_surname("Jung Hoo Lee")[0], "LEE")

    def test_hyphen_group_wins(self):
        # 'Seong' is also a surname; the hyphen proves it is the given name.
        self.assertEqual(names.split_surname("Ha-Seong Kim")[0], "KIM")

    def test_hangul_is_always_surname_first(self):
        self.assertEqual(names.split_surname("이정후")[0], "LEE")
        self.assertEqual(names.split_surname("강정호")[0], "KANG")

    def test_non_korean_name_keeps_all_tokens(self):
        cls, given = names.split_surname("Merrill Kelly")
        self.assertIsNone(cls)
        self.assertEqual(sorted(t.lower() for t in given), ["kelly", "merrill"])


class TestMatchingKeys(unittest.TestCase):
    EQUIVALENT = [
        ("Lee Jung-hoo", "Jung Hoo Lee"),
        ("Kim Ha-seong", "Ha-Seong Kim"),
        ("Ryu Hyun-jin", "Hyun-Jin Ryu"),
        ("Merrill Kelly", "Kelly, Merrill"),
    ]

    # Same player, different romanisation system or spelling choice.
    PHONETIC_ONLY = [
        ("Lee Jung-hoo", "Yi Jeong-hu"),
        ("Lee Jung-hoo", "이정후"),
        ("Hyun-Jin Ryu", "류현진"),
        ("Ha-Seong Kim", "김하성"),
        ("Shin-Soo Choo", "추신수"),
        ("Jung Ho Kang", "강정호"),
        ("Kwang-Hyun Kim", "김광현"),
        ("Seung-Hwan Oh", "오승환"),
    ]

    DIFFERENT = [
        ("Lee Jung-hoo", "Kim Ha-seong"),
        ("Hyun-Jin Ryu", "Jung Ho Kang"),
        ("Erick Fedde", "Merrill Kelly"),
    ]

    def test_strict_key_handles_order_and_hyphens(self):
        for a, b in self.EQUIVALENT:
            with self.subTest(a=a, b=b):
                self.assertEqual(names.strict_key(a), names.strict_key(b))

    def test_loose_key_handles_romanization_systems(self):
        for a, b in self.PHONETIC_ONLY:
            with self.subTest(a=a, b=b):
                self.assertEqual(names.loose_key(a), names.loose_key(b))

    def test_different_players_do_not_collide(self):
        for a, b in self.DIFFERENT:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(names.loose_key(a), names.loose_key(b))

    def test_keys_bundle_has_expected_fields(self):
        keys = names.name_keys("Jung-hoo Lee*")
        self.assertEqual(keys["name_clean"], "Jung-hoo Lee")
        self.assertIn("name_strict_key", keys)
        self.assertIn("name_loose_key", keys)


class TestKoreanNameHeuristic(unittest.TestCase):
    def test_identifies_korean_names(self):
        self.assertTrue(names.is_probably_korean_name("Lee Jung-hoo"))
        self.assertTrue(names.is_probably_korean_name("이정후"))

    def test_rejects_imports(self):
        self.assertFalse(names.is_probably_korean_name("Erick Fedde"))
        self.assertFalse(names.is_probably_korean_name("Sócrates Brito"))


if __name__ == "__main__":
    unittest.main()
