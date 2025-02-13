from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from const import sales_speed
from meanings import meanings_to_str

load_dotenv()


def zillow_to_str(row: pd.Series) -> str:

    city_mapping = {
        "CH": "Chicago (IL)",
        "NY": "New York (NY)",
        "LA": "Los Angeles (CA)",
    }

    city = city_mapping.get(row["city"], "Unknown City")
    type = "Single House" if row["single"] == 0 else "Condo/Townhouse"
    address = row["address"]
    parking = row["parking"]
    bathroom = row["bathroom"]
    bedroom = row["bedroom"]
    age = row["age"]
    living = row["living"]

    return (
        f"City: {city}, "
        f"Type: {type}, "
        f"Address: {address}, "
        f"Number of Parking Spaces: {parking}, "
        f"Number of Bathrooms: {bathroom}, "
        f"Number of Bedrooms: {bedroom}, "
        f"Age: {age} Years, "
        f"Living Space: {living} Square Meters"
    )


def words_to_str(
    city: str,
    single: int,
    path: str = "dataset/word_counts",
    percentage: int = 0.25,
    n_words: int = 10,
) -> str:

    city_mapping = {
        "CH": "Chicago (IL)",
        "NY": "New York (NY)",
        "LA": "Los Angeles (CA)",
    }
    single_mapping = {0: "Single House", 1: "Condo/Townhouse"}
    level_mapping = {0: "Fast-Selling", 2: "Slow-Selling"}

    statement = ""

    for level in [0, 2]:
        df = pd.read_csv(
            f"{path}/{percentage}/{city}_{single}_group_{level}_zscore.csv", header=None
        )
        words = df.iloc[:, 0].head(n_words).tolist()
        res = f"{level_mapping[level]} {single_mapping[single]} in {city_mapping[city]}: {words}\n"
        statement += res

    return statement


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class BinaryTOM(BaseModel):
    """Estimated time-on-market(TOM) until the property is sold."""

    TOM: Annotated[
        str,
        Field(description="Determine if the property was sold fast: 'yes' or 'no'."),
    ]
    REASON: Annotated[
        str, Field(description="Provide a brief reason for the decision.")
    ]


# basic
basic_system = f"""You are an expert realtor.
The user will provide following information about properties that already have been sold on Zillow.
1. Description (text written by the realtor)
2. Attributes (basic information such as the number of rooms, size, etc.) 
Your task is to determine whether the property was sold fast or not.
The upper limit for "fast" selling is as follows:
- For Chicago (IL), Single House: {sales_speed["CH"][0]} days
- For Chicago (IL), Condo/Townhouse: {sales_speed["CH"][1]} days
- For New York (NY), Single House: {sales_speed["NY"][0]} days
- For New York (NY), Condo/Townhouse: {sales_speed["NY"][1]} days
- For Los Angeles (CA), Single House: {sales_speed["LA"][0]} days
- For Los Angeles (CA), Condo/Townhouse: {sales_speed["LA"][1]} days
Respond with a binary score: 'yes' or 'no', and provide a brief reason (around 200 characters) for your decision.
"""

basic_template = ChatPromptTemplate.from_messages(
    [
        ("system", basic_system),
        ("human", """[Description]\n{description}\n\n[Attributes]\n{attributes}\n\n"""),
    ]
)

basic_llm = llm.with_structured_output(BinaryTOM)
basic_grader = basic_template | basic_llm


# with words
words_system = f"""You are an expert realtor.
The user will provide following information about properties that already have been sold on Zillow.
1. Description (text written by the realtor)
2. Attributes (basic information such as the number of rooms, size, etc.)
3. Discriminative Words (words commonly used in homes that sell quickly or late) 
Your task is to determine whether the property was sold fast or not.
Your task is to determine whether the property was sold fast or not.
The upper limit for "fast" selling is as follows:
- For Chicago (IL), Single House: {sales_speed["CH"][0]} days
- For Chicago (IL), Condo/Townhouse: {sales_speed["CH"][1]} days
- For New York (NY), Single House: {sales_speed["NY"][0]} days
- For New York (NY), Condo/Townhouse: {sales_speed["NY"][1]} days
- For Los Angeles (CA), Single House: {sales_speed["LA"][0]} days
- For Los Angeles (CA), Condo/Townhouse: {sales_speed["LA"][1]} days
Respond with a binary score: 'yes' or 'no', and provide a brief reason (around 200 characters) for your decision.
"""


def build_words_template(city: str, single: int):

    discriminative_words = words_to_str(city, single)
    words_template = ChatPromptTemplate.from_messages(
        [
            ("system", words_system),
            (
                "human",
                "[Description]\n{description}\n\n"
                "[Attributes]\n{attributes}\n\n"
                "[Discriminative Words]\n{words}",
            ),
        ]
    ).partial(words=discriminative_words)

    return words_template


words_llm = llm.with_structured_output(BinaryTOM)


# with words & meanings
full_system = f"""You are an expert realtor.
The user will provide following information about properties that already have been sold on Zillow.
1. Description (text written by the realtor)
2. Attributes (basic information such as the number of rooms, size, etc.)
3. Discriminative Words (words commonly used in homes that sell quickly or late)
3-1. Meanings (implications of these words for potential buyers in the context of real estate transactions) 
Your task is to determine whether the property was sold fast or not.
The upper limit for "fast" selling is as follows:
- For Chicago (IL), Single House: {sales_speed["CH"][0]} days
- For Chicago (IL), Condo/Townhouse: {sales_speed["CH"][1]} days
- For New York (NY), Single House: {sales_speed["NY"][0]} days
- For New York (NY), Condo/Townhouse: {sales_speed["NY"][1]} days
- For Los Angeles (CA), Single House: {sales_speed["LA"][0]} days
- For Los Angeles (CA), Condo/Townhouse: {sales_speed["LA"][1]} days
Respond with a binary score: 'yes' or 'no', and provide a brief reason (around 200 characters) for your decision.
"""


def build_full_template(city: str, single: int):

    discriminative_words = words_to_str(city, single)
    meanings = meanings_to_str(city, single)

    full_template = ChatPromptTemplate.from_messages(
        [
            ("system", full_system),
            (
                "human",
                "[Description]\n{description}\n\n"
                "[Attributes]\n{attributes}\n\n"
                "[Discriminative Words]\n{words}\n"
                "[Meanings]\n{meanings}",
            ),
        ]
    ).partial(words=discriminative_words, meanings=meanings)

    return full_template


full_llm = llm.with_structured_output(BinaryTOM)


if __name__ == "__main__":

    zillow = pd.read_csv("dataset/2. zillow_cleaned.csv")
    sample = zillow.iloc[0]
    print(f"\n{sample["zpid"]}: {sample["duration"]} Days")

    basic_result = basic_grader.invoke(
        input={
            "description": sample["description"],
            "attributes": zillow_to_str(sample),
        }
    )
    print(f"Basic: {basic_result}")

    # words_result = words_grader.invoke(
    #     input={
    #         "description": sample["description"],
    #         "attributes": zillow_to_str(sample),
    #     }
    # )
    # print(f"Words: {words_result}")

    # full_result = full_grader.invoke(
    #     input={
    #         "description": sample["description"],
    #         "attributes": zillow_to_str(sample),
    #     }
    # )
    # print(f"Full: {full_result}")
