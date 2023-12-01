import logging
import openai
from chat import ask
import os
import datetime as dt

import sqlalchemy as sql

from config import postgresIP, postgrsPort, postgrsUser, postgrsPass, postgrsDB
from config import arxivCategories, catNameLookup
from config import root

from utils import build_postgrs_uri

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

assert OPENAI_API_KEY is not None

if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    uri = build_postgrs_uri(postgresIP, postgrsPort, postgrsUser, postgrsPass, postgrsDB)
    engine = sql.create_engine(uri)

    currentWeekday = dt.datetime.today().weekday()
    if currentWeekday == 5:
        TDELT = 2
    elif currentWeekday == 6:
        TDELT = 3
    else:
        TDELT = 1


    with open(os.path.join(root, "summaryResults.html"), "w") as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Daily Paper Summary</title>
    <link rel="icon" href="./favicon.ico" type="image/x-icon">
<style>
.paper {
  border-radius: 25px;
  background: #cfd1d0;
  padding: 20px;
}

/* Style the tab */
.tab {
  overflow: hidden;
  border: 1px solid #ccc;
  background-color: #f1f1f1;
}

/* Style the buttons inside the tab */
.tab button {
  background-color: inherit;
  float: left;
  border: none;
  outline: none;
  cursor: pointer;
  padding: 14px 16px;
  transition: 0.3s;
  font-size: 17px;
}

/* Change background color of buttons on hover */
.tab button:hover {
  background-color: #ddd;
}

/* Create an active/current tablink class */
.tab button.active {
  background-color: #ccc;
}

/* Style the tab content */
.tabcontent {
  display: none;
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-top: none;
}
</style>
  </head>
<script>
function openCat(evt, catName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(catName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>
<body>""")
        f.write(f"<h1>Summary of papers published on {dt.datetime.today().date() - dt.timedelta(TDELT)}</h1>\
        <div class=\"tab\">")
        for cat in arxivCategories:
            f.write(f"<button class=\"tablinks\" onclick=\"openCat(event, '{cat}')\">{catNameLookup[cat]}</button>\n")
        f.write("</div>\n")
        for cat in arxivCategories:
            f.write(f"<div id=\"{cat}\" class=\"tabcontent\">\n")
            f.write(f"<h2>{catNameLookup[cat]}</h2>\n")
            with engine.connect() as conn:
                metadata = sql.MetaData()
                metadata.reflect(conn)
                arxivsummary = metadata.tables['arxivsummary']

                stmt = sql.select(arxivsummary).where(sql.and_(arxivsummary.columns.published_date == dt.datetime.today().date() - dt.timedelta(TDELT), arxivsummary.columns.subjects == f"{cat}"))
                rs = conn.execute(stmt)
                for row in rs:
                    query = f"Please summarize, in 1-2 sentences, the paper titled {row.title}."

                    f.write(f"<div class=\"paper\" id=\"{row.arxiv_id}\">\n")
                    f.write(f"<h3 class=\"ptitle\"><a href={row.url}>{row.title}</a> </h4>\n")
                    f.write(f"<h4 class=\"author_list\"> {row.author_list} </h4>\n")
                    f.write(f"<hr>\n")
                    f.write(f"<p class=\"psummary\">{ask(query)}</p>\n")
                    f.write(f"</div>\n\n")
                    f.write("<br>\n")
            f.write("</div>\n")
        f.write("</body>\n</html>")

