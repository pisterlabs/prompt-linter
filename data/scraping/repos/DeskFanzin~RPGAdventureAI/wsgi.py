##Trabalho feito por André Maurell - 142365 e Gabriel Martins - 142356
##Gerador de aventura de RPG! Completamente automático, com imagens!!!

from flask import Flask, render_template, request, redirect, session, url_for,  jsonify
import openai
import random
from openai.error import RateLimitError



## no caso
##deste trabalho as threads do gunicorn declaradas no arquivo Procfile nesta pasta. 

app = Flask(__name__)
app.secret_key = "mysecretkey"  # adicionando chave secreta para usar sessões
openai.api_key = '' ## chave da API do openAI aqui

##Rota principal, onde o usuário escolhe o nome e a aventura, e o programa gera a história.
@app.route("/", methods=("GET", "POST")) 
def index():
    session['count'] = 0
    if request.method == "POST":
            name = request.form["adventure"]
            
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=generate_prompt(name),
                temperature=1,
                max_tokens=150,
            )
            session["current_story_state"] = response.choices[0].text
            return redirect(url_for("result", result=response.choices[0].text, background=None, character=None))

    return render_template("index.html")

##Rota para gerar a imagem do personagem, usando a API do openAI.
@app.route("/character", methods=("GET", "POST"))
def character():
    if request.method == "POST":
        try:
            character_text = request.form["character_text"]
            character_image = openai.Image.create(prompt=str(character_text), n=1, size="256x256")
            session["character_image"] = character_image['data'][0]['url']
            return jsonify({"character_image": character_image['data'][0]['url']})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

##Rota para gerar a imagem do background, usando a API do openAI.
@app.route("/background", methods=("POST",))
def background():
    if request.method == "POST":
        try:
            background_text = request.form["background_text"]
            background_image = openai.Image.create(prompt=str(background_text), n=1, size="1024x1024")
            session["background_image"] = background_image['data'][0]['url']
            return jsonify({"background_image": background_image['data'][0]['url']})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

##Função de gerar imagens, usando a API do openAI.
## temos que usar     <img src="{{ result }}" alt="result" />
## para que a imagem seja renderizada na página html


##Rota para gerar a história com as opções.
@app.route("/result", methods=("GET", "POST"))
def result():
    background_image = session.get("background_image", "")

    # Pegando a URL da imagem do personagem da sessão
    character_image = session.get("character_image", None)

    if request.method == "POST":
        if session['count'] == 2:

            choice = request.form["choice"]
            current_story_state = session.get("current_story_state", "")
            new_story_state = update_story_state(current_story_state, choice)
            session["current_story_state"] = new_story_state
            try:
                response = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=generate_prompt3(),
                    temperature=0.8,
                    max_tokens=200,
                )
            except RateLimitError:
                return "<h1>Espera um pouco, você está fazendo muitas requisições!</h1> <h2> volte para a página anterior a esta e tente novamente</h2>"
            session["current_story_state"] = response.choices[0].text

          
            return redirect(url_for("ending", result=response.choices[0].text, background_image=background_image, character_image=character_image))
        else:
            choice = request.form["choice"]

            current_story_state = session.get("current_story_state", "")
            new_story_state = update_story_state(current_story_state, choice)
            session["current_story_state"] = new_story_state

            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=generate_prompt2(choice),
                temperature=0.8,
                max_tokens=200,
            )
            session["current_story_state"] = response.choices[0].text

            session['count'] += 1
            return redirect(url_for("result", result=response.choices[0].text, background_image=background_image, character_image=character_image))
        

    result = request.args.get("result")
    return render_template("result.html", result=result, background_image=background_image, character_image=character_image)

##Rota para gerar o final da história, com a imagem do vilão.
@app.route("/ending", methods=("GET", "POST"))
def ending():
    if request.method == "POST":
        try:
            if request.form["diceroll"] == "diceroll":
                return redirect(url_for("start_battle"))
        except KeyError:
            pass
    else:
        background_image = session.get("background_image", "")
        # Pegando a URL da imagem do personagem da sessão
        character_image = session.get("character_image", None)
        #criando prompt do openai para gerar a imagem do vilao
        image_prompt = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=generate_prompt_image(session['current_story_state']),
            temperature=1,
            max_tokens=200,
        )
        image_prompt = image_prompt.choices[0].text

        boss_image = openai.Image.create(prompt=str(image_prompt), n=1, size="256x256")

        session["boss_image"] = boss_image['data'][0]['url']

            
        result = request.args.get("result")
        return render_template("ending.html", result=result, background_image=background_image, character_image=character_image, boss_image=boss_image['data'][0]['url'])



@app.route("/start_battle", methods=("GET",))
def start_battle():

    # Inicialize as vidas do usuário e do chefe
    session["user_life"] = 10
    session["boss_life"] = 20
    session["user"] = 0
    session["boss"] = 0

    return redirect(url_for("battle"))


##Rota para gerar a batalha, com o dado e as vidas.
@app.route("/battle", methods=("GET", "POST"))
def battle():    
    background_image = session.get("background_image", "")
    boss_image = session.get("boss_image", "")
    character_image = session.get("character_image", "")

    user_life = session.get("user_life", 10)
    boss_life = session.get("boss_life", 20)
    user = session.get("user", 0)
    boss = session.get("boss", 0)

    if request.method == "POST":
        attack_or_defend = request.form["attack"]
        if attack_or_defend == "attack":
            # Simule o ataque do usuário com um dado de 0 a 10
            user_attack = random.randint(0, 10)
            user = user_attack
            # Reduza a vida do chefe com base no ataque do usuário
            boss_life -= user_attack
            if user_attack == 0:
                user = "Você errou o ataque!"

            # Simule o ataque do chefe com um dado de 0 a 5
            boss_attack = random.randint(0, 5)
            boss = boss_attack
            # Reduza a vida do usuário com base no ataque do chefe
            user_life -= boss_attack
            if boss_attack == 0:
                boss = "O boss errou o ataque!"
        elif attack_or_defend == "defend":
            # Simule a defesa do usuário com um dado de 0 a 5
            user_defense = random.randint(0, 8)
            user = user_defense
            # Reduza o ataque do chefe com base na defesa do usuário
            boss_attack = random.randint(0, 5) - user_defense
            boss = boss_attack
            # Reduza a vida do usuário com base no ataque do chefe
            user_life -= boss_attack
            if boss_attack == 0:
                boss = "O boss errou o ataque!"

        # Atualize as vidas na sessão
        session["user_life"] = user_life
        session["boss_life"] = boss_life
        session["user"] = user
        session["boss"] = boss

        # Verifique se alguém ganhou ou perdeu
        if user_life <= 0:
            return redirect(url_for("game_over", result="Infelizmente você acabou sucumbindo para o boss!", background_image=background_image))
        elif boss_life <= 0:
            return redirect(url_for("game_over", result="Parabéns jogador, você derrotou o boss!", background_image=background_image))

        return render_template("battle.html", user_life=user_life, boss_life=boss_life, user=user, boss=boss, background_image=background_image, boss_image=boss_image, character_image=character_image)

    return render_template("battle.html", user_life=user_life, boss_life=boss_life, user=user, boss=boss, background_image=background_image, boss_image=boss_image, character_image=character_image)

##Rota para o game over, com o final da história.
@app.route("/game_over/<result>", methods=("GET",))
def game_over(result):
    session['count'] = 0
    #resetando as variáveis de sessão
    session["character"] = None
    session["background_image"] = None
    session["character_image"] = None
    if result == "Infelizmente você acabou sucumbindo para o boss!":
        ending = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=generate_prompt_badending(),
            temperature=1,
            max_tokens=200,
        )
        ending_image = openai.Image.create(prompt=str(ending.choices[0].text), n=1, size="256x256")
        session["current_story_state"] = None
        return render_template("game_over.html", result=ending.choices[0].text, ending_image=ending_image['data'][0]['url'])
    elif result == "Parabéns jogador, você derrotou o boss!":
        ending = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=generate_prompt_goodending(),
            temperature=1,
            max_tokens=200,
        )
        ending_image = openai.Image.create(prompt=str(ending.choices[0].text), n=1, size="256x256")
        session["current_story_state"] = None
        return render_template("game_over.html", result=ending.choices[0].text, ending_image=ending_image['data'][0]['url'])


##Função para atualizar o estado da história, de acordo com a escolha do usuário.
def update_story_state(current_state, choice):

    # get the options from the string new_story_state
    
    session["option1"] = current_state.split("1-")[1].split(",")[0]
    session["option2"] = current_state.split("2-")[1].split(".")[0]
    
    #setting the current_state without the options
    session["current_state_woptions"] = current_state.split("1-")[0]

    option1_text = session.get("option1", "")
    option2_text = session.get("option2", "")

    if choice == "1":
        new_state = current_state + option1_text
    elif choice == "2":
        new_state = current_state + option2_text

    else:
        new_state = current_state

    return new_state

##Função para gerar o prompt inicial, de acordo com a aventura escolhida pelo usuário.
def generate_prompt(adventure):
    return f"""Você é um mestre de RPG e está criando uma aventura para um jogador. Ele escolhe seu nome e dá o início da aventura. Continue a história, gerando, a seu critério, entre 30 a 100 palavras e dê 2 opções do que fazer.
Nome: Gandalf, sou um mago atrás do meu chapéu.
Aventura: Você, atrás do seu chapéu há algumas semanas, finalmente achou uma pista de onde ele está. Você está em uma floresta, e vê uma caverna. Você entra na caverna e vê um dragão. 1- Lutar com o dragão, 2- Fugir da caverna.
Nome: Aragorn, sou um guerreiro atrás de uma espada mágica.
Aventura: Após você, Aragorn, sair da taverna, vai direto para a floresta, atrás de sua espada. Você encontra um esqueleto, e ele pode ter dicas de onde sua espada está. 1- Perguntar ao esqueleto, 2- Ir atrás de mais pistas.
Nome: {adventure.capitalize()}
Aventura: """

##Função para gerar o prompt da batalha, de acordo com a opção da aventura escolhida pelo usuário.
def generate_prompt2(choice):
    current_story_state = session.get("current_state_woptions", "")

    option1 = session.get("option1", "")
    option2 = session.get("option2", "")

    return f"""De acordo com sua escolha anterior, o usuário optou por fazer uma ação. Agora, continue a história, tente dar um rumo para um possível final, e sempre forneça 2 opções do que fazer. Gere entre 30 a 100 palavras.
Exemplos (se tiver mais de 100 palavras, tente gerar menos palavras na próxima vez):
Opção: 1
Aventura: {current_story_state}. 1-{option1}, 2-{option2}.
Opção: {choice.capitalize()}
Aventura:"""

##Função para gerar o prompt do final da história, de acordo com a aventura anterior do usuário.
def generate_prompt3():
    current_story_state = session.get("current_state_woptions", "")

    return f""" A história está acabando! Crie um confronto final, de acordo com a história previamente gerada, onde o usuário deverá batalhar. O vilão tera 20 de vida e deixe o usuário agir. Gere entre 30 a 100 palavras. (voce deve somente criar o confronto, não
    pode gerar o resultado.)
História antiga: Você entra nas catacumbas atrás de seu chapéu, tem muitos esqueletos no chão.
Final: Um esqueleto gigante aparece, e ele está com seu chapéu! Você tem que derrotá-lo para pegar seu chapéu de volta! O esqueleto tem 20 de vida. 
História antiga: Você entra na caverna e vê um dragão.
Final: O dragão está dormindo, e você tem que pegar sua espada de volta. Você pega sua espada e o dragão acorda! Ele tem 20 de vida.
História antiga: {current_story_state}
Final:"""

##Função para gerar o final ruim da história, de acordo com a aventura anterior do usuário.
def generate_prompt_badending():
    current_story_state = session.get("current_state_woptions", "")

    return f"""O usuário perdeu a batalha contra o chefe, Gere o final da história, de acordo com a história previamente gerada. Gere entre 30 a 100 palavras.
    História antiga: Um esqueleto gigante aparece, e ele está com seu chapéu! Você tem que derrotá-lo para pegar seu chapéu de volta! O esqueleto tem 20 de vida. 
    Final: O esqueleto te derrota e você, derrotado, foge de volta para a cidade. Você nunca mais vê seu chapéu.
    História antiga: O dragão está dormindo, e você tem que pegar sua espada de volta. Você pega sua espada e o dragão acorda! Ele tem 20 de vida.
    Final: O dragão te derrota mas você consegue fugir, e vive como um guerreiro que ainda perambula atrás de sua espada.
    História antiga: {current_story_state}
    Final:"""

##Função para gerar o final bom da história, de acordo com a aventura anterior do usuário.
def generate_prompt_goodending():
    current_story_state = session.get("current_state_woptions", "")

    return f"""O usuário ganhou a batalha contra o chefe, Gere o final da história, de acordo com a história previamente gerada. Gere entre 30 a 100 palavras.
    História antiga: Um esqueleto gigante aparece, e ele está com seu chapéu! Você tem que derrotá-lo para pegar seu chapéu de volta! O esqueleto tem 20 de vida. 
    Final: Você derrota o esqueleto e pega seu chapéu de volta! Você volta para a cidade e vive como um mago com seu querido chapéu.
    História antiga: O dragão está dormindo, e você tem que pegar sua espada de volta. Você pega sua espada e o dragão acorda! Ele tem 20 de vida.
    Final: Você derrota o dragão e pega sua espada de volta! Com ela, você se torna um guerreiro lendário da sua aldeia.
    História antiga: {current_story_state}
    Final:"""

##Função para gerar o prompt da imagem do vilão, de acordo com a aventura anterior do usuário.
def generate_prompt_image(original_prompt):
    return f"""Você recebe um texto de entrada, transforme este texto em um prompt para gerar uma imagem.
    Texto recebido: "Um dragão aparece na sua frente, ele tem 20 de vida. Você tem que derrotá-lo para pegar sua espada de volta."
    Prompt: Um dragão poderoso em uma batalha.
    Texto recebido: Você recebeu o texto: "Um esqueleto aparece na sua frente, ele tem 20 de vida. Você tem que derrotá-lo para pegar seu chapéu de volta."
    Prompt: Um esqueleto guerreiro.
    Texto recebido: {original_prompt}
    Prompt:"""



if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)
