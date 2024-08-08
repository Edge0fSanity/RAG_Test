from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer only with 'true' or 'false') Does the actual response match the expected response? 
"""



def benefir_owner():
    assert query_and_validate(
        question="Кто является бенефициарным владельцем Общества «БПС Программные продукты»?",
        expected_response="Бубнов Дмитрий Владимирович"
    )

def average_number_of_employees():
    assert query_and_validate(
        question="Какая среднесписочная численность работников Общества за 2023 год?",
        expected_response="367 человек"
    )

def share_capital():
    assert query_and_validate(
        question="Какой размер уставного капитала Общества?",
        expected_response="10 тысяч рублей"
    )

def country_risks():
    assert query_and_validate(
        question="Какие риски связаны с политической и экономической ситуацией в стране, в которой осуществляет деятельность Общество?",
        expected_response="Страновые и региональные риски, связанные с политической и экономической ситуацией, географическими особенностями, военными конфликтами, введением чрезвычайного положения, забастовками, стихийными бедствиями."
    )

def liquidity_management():
    assert query_and_validate(
        question="Какие меры принимаются Обществом для управления риском ликвидности?",
        expected_response="Общество управляет ликвидностью, поддерживая достаточные остатки денежных средств и кредитных ресурсов, регулярно мониторируя прогнозные и фактические денежные поступления и расходы, осуществляя строгий контроль за погашением дебиторской задолженности покупателями."
    )

def company_activities():
    assert query_and_validate(
        question="Какие виды деятельности осуществляет Общество «БПС Программные продукты»?",
        expected_response="Разработка компьютерного программного обеспечения, консультационные услуги в данной области и другие сопутствующие услуги."
    )

def accounting_policy_changes():
    assert query_and_validate(
        question="Какие изменения в учетной политике Общества на 2023 год были внесены?",
        expected_response="Изменения в учетной политике на 2023 год отсутствуют."
    )

def previous_director():
    assert query_and_validate(
        question="Кто руководил Обществом в течение отчетного периода (2022 год) до 07.02.2023?",
        expected_response="Пацианский Олг Владимирович."
    )

def currency_fluctuation_risks():
    assert query_and_validate(
        question="Какие риски возникают у Общества в связи с колебанием валютных курсов?",
        expected_response="Возникают большие суммы курсовых разниц, что увеличивает риск неплатежей."
    )

def short_term_liabilities():
    assert query_and_validate(
        question="Какие обязательства фиксируются как краткосрочные в бухгалтерском балансе Общества?",
        expected_response="Финансовые вложения, дебиторская и кредиторская задолженность, включая задолженность по кредитам и займам, оценочные обязательства, если срок их обращения (погашения) не превышает 12 месяцев после отчетной даты."
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.1:8b")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
