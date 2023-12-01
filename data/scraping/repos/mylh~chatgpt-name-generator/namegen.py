#!/usr/bin/env python
import os
import dns.resolver
import click
import whois
import openai
import retrying

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]


def retry_if_dns_error(exception):
    return not isinstance(exception, dns.resolver.NXDOMAIN)


@retrying.retry(
    wait_fixed=1000, stop_max_attempt_number=3, retry_on_exception=retry_if_dns_error
)
def resolve_dns_with_retry(domain, dns_servers=None):
    resolver = dns.resolver.Resolver(configure=False)
    if dns_servers:
        resolver.nameservers = dns_servers
    else:
        resolver.nameservers = ["8.8.8.8", "8.8.4.4"]
    resolver.timeout = 5
    resolver.lifetime = 5
    answers = resolver.resolve(domain)
    return answers


def clean_name_for_domain(name):
    """
    Cleans a name string so it can be used as a domain name.
    """
    # Remove any special characters and spaces
    cleaned = "".join(e for e in name if e.isalnum() or e == "-")

    # Remove leading and trailing hyphens
    cleaned = cleaned.strip("-")

    # Replace any remaining spaces with hyphens
    cleaned = cleaned.replace(" ", "-")

    # Convert to lower case
    cleaned = cleaned.lower()

    return cleaned


@click.command()
@click.option("--tld", default="com", help="Top-level domain to check against")
@click.option(
    "--prompt",
    default="Startup that helps people find the best deals on flights",
    help="Prompt to query ChatGPT for list of names",
)
@click.option("--names", default=None, help="Comma-separated list of names to check")
@click.option("--output", default=None, help="Output file to write available names to")
@click.option("--model", default="gpt-3.5-turbo-1106", help="OpenAI model to use")
@click.option(
    "--dns-server", default="8.8.8.8,8.8.4.4", help="List of DNS servers to use"
)
def check_domains(tld, prompt, names, output, model, dns_server):
    """Check availability of domains for given names and TLD."""

    if not names:
        print(f'Getting Ideas for "{prompt}"...')
        system_prompt = (
            "You are a domain name generator. You are given a startup project "
            "description and you have to generate a list of fun, memorable and "
            "interesting potential names for the project. The names should be "
            "compact and sound nice. Respond stricltly with one line of a "
            "comma-separated list of names"
        )
        # Get list of names by querying ChatGPT
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        # Extract names from OpenAI response
        names = response.choices[0].message.content.strip()
        # sometimes model returns "1. Name1\n2. Name2\n3. Name3" list, handle it
        if names.startswith("1."):
            # Split the input text into lines
            lines = names.strip().split("\n")
            # Remove item numbering and strip whitespace from each line
            cleaned_lines = [line.split(". ")[1].strip() for line in lines]
            # Join the cleaned lines into a single CSV string
            names = ",".join(cleaned_lines)

    names = names.split(",")
    names = [name.strip() for name in names]
    print(f"Generated names: {', '.join(names)}")
    names = [clean_name_for_domain(name) for name in names]
    available_domains = []
    dns_servers = [x.strip() for x in dns_server.split(",")]
    for name in names:
        domain = name + "." + tld
        click.echo(f"Checking https://{domain}")
        # Check if domain name is valid
        try:
            resolve_dns_with_retry(domain, dns_servers=dns_servers)
        except (
            dns.resolver.NoAnswer,
            dns.resolver.NoNameservers,
        ) as exc:
            click.echo(f"! Can't check domain: {exc}")
            continue
        except dns.name.EmptyLabel as exc:
            click.echo(f"! Invalid domain name: {exc}")
            continue
        except dns.resolver.NXDOMAIN:
            try:
                whois_res = whois.whois(domain)
                if whois_res.status is None:
                    available_domains.append(domain)
                    click.echo(
                        click.style("✔", fg="green", bold=True)
                        + f" Domain {domain} is available!"
                    )
                else:
                    click.echo(
                        click.style("✗", fg="red", bold=True)
                        + f" Domain {domain} is not available via WHOIS"
                    )
                continue
            except whois.parser.PywhoisError as exc:
                if "No match for" in str(exc):
                    available_domains.append(domain)
                    click.echo(
                        click.style("✔", fg="green", bold=True)
                        + f" Domain {domain} is available!"
                    )
                else:
                    click.echo(f"! WHOIS error {exc}")
                continue
        click.echo(click.style("✗", fg="red", bold=True) + " Got DNS response")

    if available_domains:
        click.echo("---\nAvailable domains:")
        for domain in available_domains:
            click.echo(domain)
        if output:
            with open(output, "a", encoding="utf-8") as out:
                out.write("\n".join(available_domains))
                out.write("\n")
    else:
        click.echo("No available domains found for the given names and TLD.")


if __name__ == "__main__":
    check_domains()  # pylint: disable=no-value-for-parameter
