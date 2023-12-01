import argparse
import subprocess
import json

def parse_git_status(output):
    lines = output.split('\n')
    changes = {'added': [], 'modified': [], 'deleted': []}
    for line in lines:
        if line.startswith('A '):
            changes['added'].append(line[2:])
        elif line.startswith('M '):
            changes['modified'].append(line[2:])
        elif line.startswith('D '):
            changes['deleted'].append(line[2:])
    return changes

def check_repository_status():
    try:
        process = subprocess.run(['git', 'status', '--porcelain'], check=True, text=True, capture_output=True)
        output = process.stdout.strip()
        if not output:
            return {"status": "clean", "recommendations": ["No changes detected."]}
        else:
            changes = parse_git_status(output)
            return {"status": "changes", "changes": changes, "recommendations": ["Uncommitted changes detected."]}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error_message": e.output.decode().strip()}

def integrate_with_openai(changes):
    # Simulated response from OpenAI based on detected changes
    if not changes:
        return {"analysis": "Your repository is clean. No action needed."}
    else:
        recommendations = []
        if changes['added']:
            recommendations.append(f"Consider creating a feature branch for these new files: {', '.join(changes['added'])}.")
        if changes['modified']:
            recommendations.append(f"Review the modifications for these files: {', '.join(changes['modified'])}.")
        if changes['deleted']:
            recommendations.append(f"Deleted files detected: {', '.join(changes['deleted'])}. Ensure they are removed intentionally.")
        return {"analysis": " ".join(recommendations)}

def create_parser():
    parser = argparse.ArgumentParser(description='GitWise - Your smart Git assistant.')
    parser.add_argument('action', choices=['check', 'recommend'], help='Action to perform: check the repository status or get AI-based recommendations')
    return parser

def user_interactive_prompt(recommendations):
    print("Based on the AI analysis, here are some actions you can take:")
    for index, recommendation in enumerate(recommendations, start=1):
        print(f"{index}. {recommendation}")
    choice = input("Would you like to act on any of these recommendations? (Enter the number or 'n' to skip): ")
    if choice.isdigit() and 0 < int(choice) <= len(recommendations):
        print(f"You have chosen to act on: {recommendations[int(choice) - 1]}")
        # Here we could add code to actually perform the actions, like creating branches, etc.
    else:
        print("No action will be taken.")

def cli_check():
    repo_status = check_repository_status()
    print(json.dumps(repo_status, indent=2))

def cli_recommend():
    repo_status = check_repository_status()
    if repo_status['status'] == 'error':
        print("Error checking repository status:", repo_status['error_message'])
    else:
        ai_analysis = integrate_with_openai(repo_status.get('changes', {}))
        print(ai_analysis['analysis'])
        user_interactive_prompt(ai_analysis['analysis'].split(". "))

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.action == 'check':
        cli_check()
    elif args.action == 'recommend':
        cli_recommend()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()