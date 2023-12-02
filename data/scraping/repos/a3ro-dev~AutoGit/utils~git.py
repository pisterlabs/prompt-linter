import os
import platform
import subprocess
import sys
import openai
import utils.keys as keys

openai.api_key = keys.api

class GitSSH:
    """
     A class responsible for the installation of Git and SSH related operations
    """
    def __init__(self, email):
        """
         Initializes the object with the email. This is called by __init__ and should not be called directly
         
         @param email - The email to set
        """
        self.email = email
        
    def install_git(self):
        """
         Install Git on the operating system based on the distribution or macOS package manager
         
         
         @return True if Git is installed False if Git is not
        """
        # Check if Git is installed
        try:
            subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as e:
            print(f"{e}\n")
            print("Git is not installed. Installing Git...")
            # Install Git based on the operating system
            # Install Git. Git. Git. Git e on Windows.
            if os.name == "nt":  # Windows
                subprocess.run(["winget", "install", "--id", "Git.Git", "-e", "--source", "winget"])
            # This function is used to install Git based on the operating system.
            if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):  # Linux and macOS
                # Check the specific Linux distribution or macOS package manager
                # Returns the distro of the current platform.
                if sys.platform.startswith("linux"):
                    distro = self.get_linux_distribution().lower()
                else:
                    distro = platform.mac_ver()[0].lower()
                # Install Git based on the Linux distribution or macOS package manager
                # This function is used to run the git install command.
                if distro in ["debian", "ubuntu"]:
                    subprocess.run(["sudo", "apt-get", "install", "-y", "git"])
                elif distro == "fedora":
                    subprocess.run(["sudo", "dnf", "install", "-y", "git"])
                elif distro == "gentoo":
                    subprocess.run(["sudo", "emerge", "--ask", "--verbose", "dev-vcs/git"])
                elif distro == "arch":
                    subprocess.run(["sudo", "pacman", "-S", "git"])
                elif distro == "opensuse":
                    subprocess.run(["sudo", "zypper", "install", "git"])
                elif distro == "mageia":
                    subprocess.run(["sudo", "urpmi", "git"])
                elif distro == "nixos":
                    subprocess.run(["nix-env", "-i", "git"])
                elif distro == "freebsd":
                    subprocess.run(["sudo", "pkg", "install", "git"])
                elif distro == "openbsd":
                    subprocess.run(["sudo", "pkg_add", "git"])
                elif distro == "alpine":
                    subprocess.run(["sudo", "apk", "add", "git"])
                elif distro == "darwin":
                    # Run the git install git and port
                    if subprocess.run(["which", "brew"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                        subprocess.run(["brew", "install", "git"])
                    elif subprocess.run(["which", "port"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                        subprocess.run(["sudo", "port", "install", "git"])
                    else:
                        print("Homebrew or MacPorts not found. Please install Git manually.")
                        return 
                else:
                    print("Unsupported Linux distribution or macOS version. Please install Git manually.")
                    return
            else:
                print("Unsupported operating system. Please install Git manually.")
                return
            
    def get_linux_distribution(self):
        """
         Get the Linux distribution. This is used to determine whether or not we are running on Linux or not.
         
         
         @return A string of the Linux distribution or " " if not
        """
        try:
            with open("/etc/os-release", "r") as f:
                lines = f.readlines()
                # Returns the ID of the first ID in the line.
                for line in lines:
                    # Returns the ID of the line.
                    if line.startswith("ID="):
                        return line.split("=")[1].strip().lower()
        except FileNotFoundError as e:
            print(e)
            

        return ""
    
    def generate_ssh_key(self):
        """
         Generate SSH key and print documentation on how to connect to GitHub. This is done by running ssh - keygen on every file
        """
        # Generate SSH key pair
        home_dir = os.path.expanduser("~")
        ssh_dir = os.path.join(home_dir, ".ssh")
        key_file = os.path.join(ssh_dir, "id_rsa.pub")
        print("Contents of .ssh directory:")
        # Prints out the files in the ssh_dir
        for file_name in os.listdir(ssh_dir):
            print(f">-+-< {file_name} >-+-<")
        subprocess.run(["ssh-keygen", "-t", "rsa", "-b", "4096", "-C", self.email])

        # Print SSH key 
        with open(key_file, "r") as f:
            ssh_key = f.read()
        print("SSH key:")
        print(ssh_key)

    
        # Print documentation on how to connect to GitHub
        print("Documentation:")
        print("1. Copy the SSH key above.")
        print("2. Go to your GitHub account settings.")
        print("3. Click on 'SSH and GPG keys'.")
        print("4. Click on 'New SSH key' or 'Add SSH key'.")
        print("5. Paste the copied SSH key into the 'Key' field.")
        print("6. Provide a suitable title for the key.")
        print("7. Click 'Add SSH key' or 'Add key'.")
        confirmation: str = str(input("Are you done with these steps?: [y/n]"))
        # Confirm the user is confirmed.
        if confirmation == "y":
            # Check if an existing SSH connection to GitHub exists
            github_host = "github.com"
            ssh_config_file = os.path.join(ssh_dir, "config")
            with open(ssh_config_file, "r") as f:
                ssh_config = f.read()
            # If there is an existing SSH connection to GitHub
            if github_host in ssh_config:
                print("Existing SSH connection to GitHub:")
                print(ssh_config)
            subprocess.run(["ssh", "-T", "git@github.com"])
    
        else:
            issue: str = str(input("What is the issue that you are you facing?: "))
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a github expert and a DevOps enthusiast, your name is AutoGit and you help people setup git and github."},
                {"role": "user", "content": issue}
            ],
        )

        assistant_reply = response.choices[0].message.content.strip() #type: ignore
        print(assistant_reply)
        confirmation: str = str(input("Is Your Issue Solved? [y/n]: ")) 
        # Confirm the user is confirmed.
        if confirmation == "y":
            # Check if an existing SSH connection to GitHub exists
            github_host = "github.com"
            ssh_config_file = os.path.join(ssh_dir, "config")
            with open(ssh_config_file, "r") as f:
                ssh_config = f.read()
            # If there is an existing SSH connection to GitHub
            if github_host in ssh_config:
                print("Existing SSH connection to GitHub:")
                print(ssh_config)
            subprocess.run(["ssh", "-T", "git@github.com"])
        else:
            print("Issue Still Not Solved? Search for the issue on Google.")
            sys.exit()

# This is the main entry point for the main module.
if __name__ == "__main__":
    # Example usage
    email = input("Enter Your Email: ")
    git = GitSSH(email=email)
    git.install_git()
    git.generate_ssh_key()
