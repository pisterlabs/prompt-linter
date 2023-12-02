import argparse

from episode_parser import parse_episode
from vertexai_players import VertexAITextPlayer
from model import Player
from openai_players import GPTPlayer

parser = argparse.ArgumentParser(
    description='Play "Auksinis protas" episode by LLM')
parser.add_argument(
    '-e', '--episode', required=True, type=str, help='Path to the episode text file')
parser.add_argument(
    '-r', '--round', required=True, type=int, choices=[1, 2, 3, 4], help='Round to play')
parser.add_argument(
    '-l', '--llm', required=True, type=str, choices=['gpt-3.5', 'gpt-4', 'palm'], help='LLM to play with')


def resolve_player(llm: str) -> Player:
    if llm == 'gpt-3.5':
        return GPTPlayer(model_name='gpt-3.5-turbo')
    elif llm == 'gpt-4':
        return GPTPlayer(model_name='gpt-4')
    elif llm == 'palm':
        return VertexAITextPlayer()
    raise ValueError(f"Unknown LLM '{llm}'")


def main(episode_file: str, round: int, llm: str) -> None:
    print(f"Playing episode file '{episode_file}' round '{round}' with LLM '{llm}'")

    episode = parse_episode(episode_file)
    player = resolve_player(llm)

    if round == 1:
        correct_answers = 0
        for question in episode.round1:
            print(f"Question: {question.question}")
            print(f"Options: {', '.join(question.options)}")
            print(f"Expected answer: {question.answer}")
            llm_answer = player.play_round1(question)
            correct = llm_answer.answer == question.answer
            print(f"LLM answer: {llm_answer.answer}. Correct: {correct}.")
            correct_answers += 1 if correct else 0
        print(f"Correct answers: {correct_answers}/{len(episode.round1)}")
    elif round == 2:
        score = 0
        for question in episode.round2:
            print(f"Question: {question.question}")
            print(f"Expected answer: {question.answer}")
            used_hints = []
            for hint in question.hints:
                used_hints.append(hint)
                llm_answer = player.play_round2(question, used_hints)
                correct = llm_answer.answer.lower() == question.answer.lower()
                question_score = 5 - (len(used_hints)) if correct else 0
                print(
                    f"Expected answer: '{question.answer}' LLM answer: '{llm_answer.answer}' with {len(used_hints)} hints. "
                    f"Correct: {correct}. Score: {question_score}.")
                if correct:
                    score += question_score
                    break
        print(f"Episode score: {score}")
    elif round == 3:
        score = 0
        for question in episode.round3:
            print(f"Question: {question.question}")
            print(f"Choices: {', '.join(question.choices)}")
            for answer in question.answers:
                query, correct_answer = answer
                llm_answer = player.play_round3(question, query)
                correct = llm_answer.answer.lower() == correct_answer.lower()
                score += 1 if correct else 0
                print(f"Query: '{query}' Expected answer: '{correct_answer}' LLM answer: '{llm_answer.answer}' "
                      f"Correct: {correct}. Current score: {score}")
    elif round == 4:
        score = 0
        for question in episode.round4:
            print(f"Question: {question.question}")
            llm_answer = player.play_round4(question)
            correct = llm_answer.answer.lower() == question.answer.lower()
            score += 1 if correct else 0
            print(f"Expected answer: '{question.answer}' LLM answer: {llm_answer.answer}. Correct: {correct}.")
        print(f"Episode score: {score}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.episode, args.round, args.llm)
