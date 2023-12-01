import openai
import os


# OpenAI KEY
OPENAI_API_KEY = "KEY"

# txt 경로 설정
source_folder = '/abstract'
combined_file_path = '/datas/output.txt'


def read_txt_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f'Error: {e}')
        return None


def create_response_txt_files(source_folder, combined_file_path):
    try:
        combined_content = ''

        for filename in os.listdir(source_folder):
            if filename.endswith('.txt'):
                source_file_path = os.path.join(source_folder, filename)
                abstract = read_txt_content(source_file_path)

                if abstract is not None:
                    combined_content += get_gpt_response(abstract)

        with open(combined_file_path, 'a') as combined_file:
            combined_file.write(combined_content)
    except Exception as e:
        print(f'Error: {e}')


def get_gpt_response(text):
    # openai API 키 인증
    openai.api_key = OPENAI_API_KEY
    model = "gpt-3.5-turbo"
    
    query = text + "Please create 1 questions and answers related to this content."

    messages = [
        {"role": "system", "content": "You are a robot that creates question and answer pairs."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": "Q: How does the use of a fishbone pattern in the printed circuit board Rogowski coil contribute to its effectiveness as a current sensor for high-voltage power modules?\nA: The fishbone pattern in the Rogowski coil optimizes magnetic field coupling, ensuring an efficient and compact design that enhances sensitivity and accuracy in measuring currents within high-voltage power modules. This layout allows for a more ideal Rogowski coil, enabling cost-effective and precise monitoring and protection of power modules and packages."},
        {"role": "assistant", "content": "Q: How does the fishbone pattern of the printed circuit board Rogowski coil contribute to its effectiveness as a current sensor in high-voltage applications?\nA:The fishbone pattern in the printed circuit board Rogowski coil enhances its effectiveness as a current sensor in high-voltage applications by maximizing the magnetic field coupling with the conductor carrying the high current. This specific pattern allows for a more compact and efficient design, making the coil small, thin, and cost-effective. The shape and layout of the fishbone pattern are strategically designed to capture and integrate the magnetic field induced by the current, resulting in an accurate measurement of the current passing through the monitored conductor. This design also facilitates noise reduction under high-voltage and high-current conditions, making it an ideal choice for monitoring and protecting high-voltage power modules and packages."},
        {"role": "assistant", "content": "Q: What are the key features that make the printed circuit board Rogowski coil an ideal current sensor for high-voltage power modules?\nA: The printed circuit board Rogowski coil stands out as an ideal current sensor for high-voltage power modules due to several key features. Firstly, its compact and thin design, along with being cost-effective, make it highly practical for integration into power modules and packages. The fishbone pattern in its design further enhances its efficacy as a Rogowski coil. Additionally, for noise reduction in high-voltage and high-current environments within a module, shield layers and coaxial connectors have been strategically utilized. Furthermore, the development of a fast simulation tool has been pivotal in optimizing the main coil pattern to suit arbitrary printed circuit board geometries, especially in confined spaces, ensuring optimal performance and functionality."},
        {"role": "assistant", "content": "Q: How does the fishbone pattern of the printed circuit board Rogowski coil contribute to its functionality and efficiency as a current sensor in high-voltage power modules?\nA: The fishbone pattern in the printed circuit board Rogowski coil enhances its functionality and efficiency as a current sensor in high-voltage power modules in several ways. Firstly, the unique fishbone pattern allows for a compact and efficient layout of the coil on the printed circuit board, enabling a small and thin form factor which is ideal for space-constrained applications. Additionally, this pattern optimizes the magnetic field coupling, resulting in improved sensitivity and accuracy in current measurements. The design of the fishbone pattern also helps to reduce noise in high-voltage and high-current conditions commonly encountered in power modules. The arrangement minimizes electromagnetic interference and inductive coupling with nearby conductors, leading to a cleaner and more precise current measurement. Moreover, the integration of shield layers and a coaxial connector further aids in noise reduction, ensuring reliable performance even in challenging electrical environments. Furthermore, the ability to customize and optimize the main coil pattern using a fast simulation tool is a significant advantage. This tool allows for tailoring the coil geometry to specific and limited spaces within the module, maximizing the current sensing capability while adhering to space constraints. In summary, the fishbone pattern, noise reduction strategies, and simulation tools collectively make this printed circuit board Rogowski coil an almost ideal solution for monitoring and protecting high-voltage power modules."},
        {"role": "assistant", "content": "Q: What are the key advantages of employing shield layers and a coaxial connector in the design of the printed circuit board Rogowski coil for current monitoring and protection of high-voltage power modules?\nA: The inclusion of shield layers and a coaxial connector in the printed circuit board Rogowski coil design provides significant advantages. Shield layers help mitigate noise interference that often occurs in high-voltage and high-current environments. They act as a barrier, reducing external electromagnetic interference and enhancing the accuracy and reliability of current measurements. The coaxial connector ensures a secure and efficient electrical connection, further minimizing noise and signal degradation. Overall, these design elements contribute to the coil's precision and effectiveness in monitoring and protecting high-voltage power modules."}
    ]

    # ChatGPT API 호출
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    return response['choices'][0]['message']['content']


create_response_txt_files(source_folder, combined_file_path)
