"""You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:""""""You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:""""""Between >>> and <<< are the raw search result text from google search html page.
Extract the answer to the question '{query}'. Please cleanup the answer to remove any extra text unrelated to the answer.

Use the format
Extracted: answer
>>> {output} <<<
Extracted:"""