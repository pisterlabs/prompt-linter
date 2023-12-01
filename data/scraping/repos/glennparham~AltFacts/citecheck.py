!pip install anthropic
!pip install textract
!pip install python-magic

import anthropic
from requests import get
from bs4 import BeautifulSoup as bs
from re import sub as resub, match as rematch, finditer
from textract import process  # everything to text
from urllib.parse import urlparse
import os  # basename etc
import mimetypes  # to get file extension from HTTP content-type
import magic  # to guess content-type when no content-type or extension


llm = anthropic.Client(api_key=  # Claude (version 2) API key 
                       '[secret]')

def claude(prompt):  # get a response from the Anthropic Claude v.2 LLM with the 100,000 token context window
  return llm.completions.create(model='claude-2', temperature=0.0,  # temperature zero for quasi-reproducability
      prompt=f'{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}',
      max_tokens_to_sample=5000, stop_sequences=[anthropic.HUMAN_PROMPT]  ### ??? is 5000 tokens a good maximum output size?
      ).completion.strip()  # unpack response


def textarticlewithrefs(title):
  # get English Wikipedia article in plain text but with numbered references including link URLs

  resp = get('https://en.wikipedia.org/w/api.php?action=parse&format=json&page='
             + title).json()
  if 'error' in resp:
    raise FileNotFoundError(f"'{ title }': { resp['error']['info'] }")

  html = resp['parse']['text']['*'] # get parsed HTML

  if '<div class="redirectMsg"><p>Redirect to:</p>' in html: # recurse redirects
    return textarticlewithrefs(resub(r'.*<ul class="redirectText"><li><a'
        + ' href="/wiki/([^"]+)"[^\0]*$', '\\1', html))

  cleantitle = resp['parse']['title'] # fixes urlencoding and unicode escapes
  try:
    asplit = html.split('<ol class="references">')
    body = '\n\n'.join(asplit[:-1])
    refs = asplit[-1] # ignore notes section
  except:
    body = html; refs = ''

  b = resub(r'\n<style.*?<table [^\0]*?</table>', '\n', body) # rm boxes
  b = resub(r'<p>', '\n<p>', b) # newlinees between paragraphs
  b = resub(r'(</table>)\n', '\\1 \n', b) # space after amboxes
  b = resub(r'(<span class="mw-headline" id="[^"]*">.+?)(</span>)',
               '\n\n\\1:\\2', b) # put colons after section headings
  b = resub(r'([^>])\n([^<])', '\\1 \\2', b) # merge non-paragraph break
  b = resub(r'<li>', '<li>* ', b) # list item bullets for beautifulsoup
  b = resub(r'(</[ou]l>)', '\\1\n\n<br/>', b) # blank line after lists
  b = resub(r'<img (.*\n)', '<br/>--Image: <img \\1\n<br/>\n', b) # captions
  b = resub(r'(\n.*<br/>--Image: .*\n\n<br/>\n)(\n<p>.*\n)',
            '\\2\n<br/>\n\\1', b) # put images after following paragraph
  b = resub(r'(role="note" class="hatnote.*\n)', '\\1.\n<br/>\n', b) # see/main
  b = resub(r'<a rel="nofollow" class="external text" href="(http[^"]+)">(.+?)</a>',
            '\\2 [ \\1 ]', b) # extract external links as bracketed urls

  b = bs(b[b.find('\n<p>'):]).get_text(' ') # to text; lead starts with 1st <p>
  b = resub(r'\s*([?.!,):;])', '\\1', b) # various space cleanups
  b = resub(r'  *', ' ', resub(r'\( *', '(', b)) # rm double spaces and after (
  b = resub(r' *\n *', '\n', b) # rm spaces around newlines
  b = resub(r'[ \n](\[\d+])', '\\1', b) # rm spaces before inline refs
  b = resub(r' \[ edit \]\n', '\n', b).strip() # drop edit links
  b = resub(r'\n\n\n+', '\n\n', b) # rm vertical whitespace

  r = refs[:refs.find('\n</ol></div>')+1] # optimistic(?) end of reflist
  r = resub(r'<li id="cite_note.*?-(\d+)">[^\0]*?<span class=' # enumerate...
            + '"reference-text"[^>]*>\n*?([^\0]*?)</span>\n?</li>\n',
           '[\\1] \\2\n', r) # ...the references as numbered seperate lines
  r = resub(r'<a rel="nofollow" class="external text" href="(http[^"]+)">(.+?)</a>',
            '\\2 [ \\1 ]', r) # extract external links as bracketed urls

  r = bs(r).get_text(' ') # unHTMLify
  r = resub(r'\s([?.!,):;])', '\\1', r) # space cleanups again
  r = resub(r'  *', ' ', '\n' + r) # rm double spaces, add leading newline
  r = resub(r'\n\n+', '\n', r) # rm vertical whitespace
  r = resub(r'(\n\[\d+]) [*\n] ', '\\1 ', r) # multiple source ref tags
  r = resub(r'\n ', '\n     ', r) # indent multiple source ref tags

  refdict = {} # refnum as string -> (reftext, first url)
  for ref in r.split('\n'):
    if len(ref) > 0 and ref[0] == '[':
      rn = ref[1:ref.find(']')] # reference number as string
      reftext = ref[ref.find(']')+2:] # omit number in text for refdict
      if '[ http' in reftext: # has a url in it
        firsturl = reftext[reftext.find('[ http')+2:reftext.find(' ]')]
        refdict[rn] = (reftext, firsturl)

  return cleantitle + '\n\n' + b + r, refdict


def verifyrefs(article):  # Wikipedia article title
  atext, refs = textarticlewithrefs(article)
  title = atext[:atext.find('\n')]
  print('Trying to verify references in article:', title)

  for par in atext.split('\n'):
    if par == 'References:' or rematch('\[\d+] [^[].+$', par):
      continue # ignore references section of article

    for m in list(finditer(r'\[\d+]', par)):  # iterate through all the references
      refnum = par[m.start()+1:m.end()-1]
      excerpt = par[:m.end()]

      # TODO: remove anything up to and including prior citations of same refnum
      # maybe remove all previous sentences ending with a reference?

      if refnum in refs:
        [reftext, url] = refs[refnum]
        print('\n Checking ref [' + refnum + '] in:', excerpt)
        print('  Reference text:', reftext)

        try:
          # Fetch the content from the URL
          response = get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; ' +
                        'Intel Mac OS X 10.15; rv:84.0) Gecko/20100101 Firefox/84.0'},
                        stream=True)  # streaming saves memory even writing all content to file
        except Exception as err:
          print('  Failed to fetch', url, 'due to', str(err))
          continue

        # Check if the request was successful
        if response.status_code == 200:

          # Get the content-type from the response headers
          content_type = response.headers['content-type']
          # Use the mimetypes library to guess the extension based on the content-type
          mime_extension = mimetypes.guess_extension(content_type)

          # Parse the URL to get the extension
          url_extension = os.path.splitext(url)[1]

          if mime_extension:  # preferred
            filename = "download" + mime_extension
            guessext = False
          elif url_extension:  # 2nd best
            filename = "download" + url_extension
            guessext = True  # because urls lie, check the contents
          else:  # must examine contents
            filename = "download"
            guessext = True

          with open(filename, 'wb') as f:
            f.write(response.content)

          if guessext:
            file_type = magic.from_file(filename, mime=True)
            guessed_extension = mimetypes.guess_extension(file_type)
            if guessed_extension:  # Rename the file with the guessed extension
              new_filename = filename + guessed_extension
              os.rename(filename, new_filename)
              filename = new_filename

          # use textract to process the file
          try:
            pagetext = process(filename).decode('utf-8')
          except:  # extension not supported
            pagetext = str(response.content)  # well, we tried

          os.unlink(filename)  # immediately delete it because there will be many

          print('  Fetching', url, 'returned', len(pagetext), 'characters.')
          if len(pagetext) < 1000:
            print('   WARNING: less than 1000 characters.')

        else:  # response.status != 200 OK
          print('  Failed to fetch', url, 'due to response status:', response.status_code)
          continue

        # need to chunk > ~350,000 chars
        window_size = 350000
        num_windows = len(pagetext) // window_size

        # Loop over the windows
        for i in range(num_windows + 1):
          # Calculate the start and end indices for the current window
          start = max(0, i * window_size - 1000)  # 1000 character overlapping windows
          end = start + window_size

          # Extract the text for the current window
          window_text = pagetext[start:end]

          # Construct the prompt
          prompt = ( 'Can the following excerpt from the Wikipedia article "'
                    + title + '" be verified by its reference [' + refnum + ']?'
                    + '\n\nThe excerpt is: ' + excerpt + '\n\nAnswer either'
                    + ' "YES: " followed by the sentence of the source text'      
                    + ' confirming the excerpt, or "NO: " followed by the reason' 
                    + ' that it does not. The source text for reference ['        
                    + refnum + '] (' + reftext.strip() + ') is:\n\n'
                    + window_text + '\n\n[End of source text]\n' )

          # Get the response
          response = resub(r'\s+', ' ', claude(prompt))
          print('  processed chunk', i + 1, 'of', num_windows + 1, 'chunks of', 
                len(pagetext), 'character source text')
        
          # If the response starts with "YES: ", break the loop
          if response.upper().startswith("YES") or "YES: " in response:
            print('  SUCCESSFULLY VERIFIED!')
            break
        
        print('  response:', response)

      else:
                            
        print('  reference [' + refnum + '] has no URL')


#print(textarticlewithrefs('Russia')[0])  ### to test the article text & URLs extractor
                            
#verifyrefs('Kadyrovites')  ### to test the main routine of the Wikipedia article
