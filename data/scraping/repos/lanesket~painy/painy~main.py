import argparse
import openai
import os
from painy import console
from painy.comment import get_commmit_message, comment_interactive
from painy.enums import Action
from painy.git import commit, get_changed_files, get_diff_str
from painy.utils import print_commit_message
from painy.managers import ConfigManager, RulesManager


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        exit(1)

    openai.api_key = api_key
    
    parser = argparse.ArgumentParser(
        prog="Painy",
        description="A tool to help you write better commit messages."
    )
    
    parser.add_argument("action", choices=["comment", "commit", "config", "rules"], help="The action to perform")
    parser.add_argument("--check-all", action="store_true", help="Check all files previously registered in git, not just staged ones")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-s", "--set", nargs=2, help="Set a config value. Example: painy config --set use_commit_history_style True")
    parser.add_argument("-g", "--get", help="Get a config value. Example: painy config --get use_commit_history_style")
    parser.add_argument("-a", "--add", help="Add a rule. Example: painy rules --add '<string>'")
    parser.add_argument("-r", "--remove", help="Remove a i-th rule from the list. Example: painy rules --remove <int>")
    
    args = parser.parse_args()
    
    action = args.action if args.action is not None else Action.COMMENT.value
    
    staged = not args.check_all
    interactive = args.interactive
    
    if action == Action.CONFIG.value:
        if args.set is not None and len(args.set) == 2:
            key, value = args.set
            console.print(f"[green]Setting config value:[/green] {key} = {value}")
                        
            config_manager = ConfigManager()
            config_manager.set_option(key, value)
        elif args.get is not None:
            key = args.get
            console.print(f"[green]Getting config value:[/green] {key}")
            
            config_manager = ConfigManager()
            value = config_manager.get_option(key)
            
            console.print(f"[green]Value:[/green] {value}") 
    
        return
    
    if action == Action.RULES.value:
        rules_manager = RulesManager()
        config_manager = ConfigManager()

        if args.add is not None:
            rule = args.add
            console.print(f"[green]Adding rule:[/green] {rule}")
            
            rules_manager.add_rule(rule)
        elif args.remove is not None:
            i = int(args.remove)
            if i <= 0:
                console.print("[red]Error: i must be greater than 0[/red]")
                return
            
            console.print(f"[green]Removing rule:[/green] {i}")
            try:
                rules_manager.remove_rule(i - 1)
            except IndexError:
                length = rules_manager.get_length()
                console.print(f"[red]Error: i must be less than or equal to {length}[/red]")
        else:
            rules = rules_manager.get_rules(config_manager.config_dict)
            rules_str = '\n'.join(rules)
            console.print(f"[green]Rules:[/green]\n{rules_str}")
        
        return
    
    try:
        changed_files = get_changed_files(staged)
        console.print(f"[green]Changed files:[/green] {', '.join(changed_files)}")
        
        diff_str = get_diff_str(changed_files)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        return
    
    if action == Action.COMMENT.value:
        msg = get_commmit_message(diff_str)
        print_commit_message(console, msg)
        
        if interactive:
            msg = comment_interactive(msg, diff_str)
            
    elif action == Action.COMMIT.value:
        msg = get_commmit_message(diff_str)
        print_commit_message(console, msg)
         
        if interactive:
            msg = comment_interactive(msg, diff_str)
            
            option = console.input("Do you want to commit with this message? [green]y[/green]/[red]n[/red]: ")
            
            if option.lower() in ["y", "yes"]:
                commit(msg)
        else:
            print_commit_message(console, msg)
            commit(msg)
        

if __name__ == "__main__":
    main()