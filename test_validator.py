import unittest

from validators import parse_answer, validate_answer


class TestAnswerValidator(unittest.TestCase):

    def test_single_letter(self):
        self.assertEqual(parse_answer("B"), "B")

    def test_answer_label(self):
        self.assertEqual(parse_answer("Answer: C"), "C")

    def test_lowercase_answer(self):
        self.assertEqual(parse_answer("answer: a"), "A")

    def test_invalid_long_answer(self):
        self.assertIsNone(
            parse_answer("The answer is B because the CPU processes instructions.")
        )

    def test_empty_answer(self):
        self.assertIsNone(parse_answer(""))

    def test_correct_answer(self):
        result = validate_answer("Answer: B", "B")

        self.assertTrue(result["format_valid"])
        self.assertTrue(result["is_correct"])

    def test_wrong_answer(self):
        result = validate_answer("Answer: C", "B")

        self.assertTrue(result["format_valid"])
        self.assertFalse(result["is_correct"])

    def test_invalid_format(self):
        result = validate_answer("Maybe B", "B")

        self.assertFalse(result["format_valid"])
        self.assertFalse(result["is_correct"])


if __name__ == "__main__":
    unittest.main()