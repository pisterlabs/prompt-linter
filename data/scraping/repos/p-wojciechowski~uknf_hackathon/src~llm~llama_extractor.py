import guidance
from guidance import one_or_more, select, gen, models
import os
from llm.few_shot_prompts import FEW_SHOT_PROMPTS


PATH = os.path.join("llm", "q4_llama7b", "ggml-model-q4_0.gguf")
llama2 = models.LlamaCpp(PATH, n_gpu_layers=16)


@guidance(stateless=True)
def number(lm):
    """model generuje liczbę"""
    n = one_or_more(select(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
    return lm + n


def cast_to_number_or_none(text: str) -> float or None:
    """zamienia tekst na liczbę lub None, jeśli tekst to 'None'"""
    if text == "None":
        return None
    else:
        return float(text)


def get_deposit_amounts_from_llama(text_to_extract: str) -> tuple:
    """zwraca minimalną i maksymalną kwotę depozytu z tekstu
    na podstawie skwantyzowanego modelu Llama2-7B i kilku przykładów (few-shot learning)"""
    final_prompt = FEW_SHOT_PROMPTS["deposit_amounts"].replace("\n", " ") + "saldo: " + text_to_extract + " "
    lm = (
        llama2
        + final_prompt
        + "minimalna kwota: "
        + select([number(), "None"], name="min_deposit")  # liczba oznaczająca minimalną kwotę depozytu lub None
        + " maksymalna kwota: "
        + select([number(), "None"], name="max_deposit")  # liczba oznaczająca maksymalną kwotę depozytu lub None
        + gen(stop="###")
    )
    min_deposit = cast_to_number_or_none(lm["min_deposit"].strip())
    max_deposit = cast_to_number_or_none(lm["max_deposit"].strip())

    return min_deposit, max_deposit


def get_duration_from_llama(text_to_extract: str) -> int:
    """zwraca czas trwania depozytu z tekstu
    na podstawie skwantyzowanego modelu Llama2-7B i kilku przykładów (few-shot learning)"""
    final_prompt = FEW_SHOT_PROMPTS["duration"].replace("\n", " ") + "okres trwania: " + text_to_extract + " "
    lm = (
        llama2
        + final_prompt
        + "czas depozytu: "
        + select([number()], name="duration")  # liczba oznaczająca czas trwania depozytu
        + select([" L", " M", " T", " D"], name="duration_unit")  # liczba oznaczająca jednostkę czasu
        + gen(stop="###")
    )

    duration = int(cast_to_number_or_none(lm["duration"].strip()))
    duration_unit = lm["duration_unit"].strip()
    n_days = get_n_days_in_unit(duration_unit)
    duration_in_days = duration * n_days
    return duration_in_days


def get_n_days_in_unit(unit_name):
    """zwraca liczbę dni w jednostce czasu"""
    if unit_name == "L":
        return 365
    elif unit_name == "M":
        return 30
    elif unit_name == "T":
        return 7
    elif unit_name == "D":
        return 1
    else:
        raise ValueError("Nieznana jednostka czasu")


if __name__ == "__main__":
    # a = get_deposit_amounts_from_llama("do 100 tys. zł")
    a = get_duration_from_llama("lokata na 7 dni")
    print(a)
