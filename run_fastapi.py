import requests


def main():
    url = "http://127.0.0.1:8000/uploadfiles/"

    files = [
        (
            "files",
            (
                "./docs/PSLE-Challenging-Math-Questions_pg2.pdf",
                open("./docs/PSLE-Challenging-Math-Questions_pg2.pdf", "rb"),
            ),
        ),
        ("files", ("./docs/PSLE-pg2_q1.png", open("./docs/PSLE-pg2_q1.png", "rb"))),
    ]

    question = "You are a math teacher. Please give a step by step explanation to the provided math questions to a 8 years old kid."

    response = requests.post(
        url=url,
        files=files,
        data={
            # "text":"",
            "question": question
        },
    )
    print(response.json())


if __name__ == "__main__":
    main()
