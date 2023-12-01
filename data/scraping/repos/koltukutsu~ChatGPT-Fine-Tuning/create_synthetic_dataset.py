import os
import sys
import openai
import pandas as pd


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    l_ages = [age for age in range(18, 22)]
    l_genders = ["erkeğim", "kadınım"]
    l_activities = [
        "Bir kitap okumak",
        "Film izlemek",
        "Müzik dinlemek",
        "Bir arkadaşla buluşmak",
        "Yeni bir tarif denemek",
        "Bahçe işleri yapmak",
        "Sanat yapmak",
        "Yeni bir dil öğrenmek",
        "Gökyüzünü izlemek",
        "Yeni bir spor denemek",
        "Fotoğraf çekmek",
        "Yeni bir hobi edinmek",
        "Bir oyun oynamak",
        "Gönüllü çalışmak",
        "Yeni bir şehir keşfetmek",
        "Atölye çalışmalarına katılmak",
        "Güzel bir yemek yapmak",
        "Bir müze ziyareti yapmak",
        "Ressamlık yapmak",
    ]
    temperatures = {0: 1, 1: 0.85, 2: 1}
    max_token = 900
    iteration = 1


    df = pd.DataFrame(
    )

    total = len(l_ages) * len(l_genders) * len(l_activities) * iteration
    counter = 0
    print("Calculating the parameters...")
    print("Total number of data: ", total)
    print(
        "Total number of tokens: ",
        max_token * total, " Cost of training: ",
        max_token * total / 1000 * 0.12,
    )
    choice = input("Press 0 to continue, otherwise close the app...\n> ")
    if choice == "0":
        print("\nStarting to create the dataset...")
        for age in l_ages:
            for gender in l_genders:
                for activity in l_activities:
                    for i in range(iteration):  ## iteration times
                        counter += 1
                        print(f"{counter}/{total}")
                        prompt = "Ben bir {gender} ve yaşım {age}. Boş zamanım var ve boş zamanımı {activity} için kullanmak istiyorum. Yaşımı ve cinsiyetimi göz önüne alarak bana bu konuyla ilgili 2 öneride bulunup, bu önerilerin açıklamasını yapıp aynı zamanda bana bir maliyet analizi yapabilir misin? Açıklamalar kısa olsun."
                        sub_prompt = "{age} yaşında bir {gender}. {activity} istiyorum."
                        
                        prompt = prompt.format(
                            gender=gender, age=age, activity=activity
                        )
                        sub_prompt = sub_prompt.format(
                            age=age, gender=gender, activity=activity
                        )
                        print(sub_prompt)
                        # print(prompt)
                        # print("----------------")
                        
                        response = openai.Completion.create(
                            model="text-davinci-003",
                            prompt=prompt,
                            temperature=temperatures[i],
                            max_tokens=max_token,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                        finish_reason = response["choices"][0]["finish_reason"]
                        response_txt = response["choices"][0]["text"]
                        
                        finish_reason = "stop"
                        link = f"youtube.com/kodzillaistanbul/{age}/{gender}/{activity}"
                        new_object = {
                            "age": age,
                            "gender": gender,
                            "activity": activity,
                            "prompt": prompt,
                            "sub_prompt": sub_prompt,
                            "response_txt": response_txt,
                            "finish_reason": finish_reason,
                            "link":link
                        }
                        new_row = pd.DataFrame([new_object])
                        df = pd.concat([df, new_row], axis=0, ignore_index=True)
                        df.to_excel("../datasets/generated_dataset.xlsx")
        print("Saved to synthetic_dataset.xlsx")

    else:
        print("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
