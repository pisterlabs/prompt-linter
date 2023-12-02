import xml.etree.ElementTree as ET
import sys, getopt
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from util import num_tokens_from_string
import shutil
import re

alphabets=['-','a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p']

def main(argv):
  print('running main')
  if os.path.exists('output'):
      shutil.rmtree('output')

  #os.removedirs('output')
  os.makedirs('output', exist_ok=True)
  inputfile = ''
  outputfile = ''
  split = True
  useLangchain = False
  opts, args = getopt.getopt(argv,'hi:o:s:l',['ifile=','ofile=','split=','langchain'])
  for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ('-i', '--ifile'):
         inputfile = arg
      elif opt in ('-l', '--langchain'):
         useLangchain = True
      elif opt in ('-o', '--ofile'):
         outputfile = arg 
      elif opt in ('-s', '--split'):
         split = arg in (True, False)
  print ('Input file is', inputfile)
  print ('Output file is', outputfile)
  if split: 
     print('splitting to multiple files (ignored for langchain split)')
  print ('useLangchain is', useLangchain)
  tree = ET.parse(inputfile)
  root = tree.getroot()
  if useLangchain: 
     print('using langchain to split')
  print('OUTPUT')
  print('====================')
  #print(root.tag)
  section = 1
  filetext = ''
  files = []
  for child in root:
     ## main title of the xml document
     if child.tag == 'title':
        filetext += '# ' + child.text
     ## all subsections are topics
     if child.tag == 'topic':
        subsection = 1
        for nested in child:
           if nested.tag == 'title':
              #print(nested.text)
              filetext += '\n## ' + nested.text
           #print(nested.tag, nested.attrib)
           #print(get_text(nested))
           #print(nested.text)
           if nested.tag == 'body':
              for elem in nested:
                 #print(elem.tag, elem.attrib)
                 if elem.tag == 'p':
                    txt = get_text(elem)
                    filetext += '\n' + txt
                    files.append(filetext)
           if nested.tag == 'topic':
             topic = get_topic(nested, '{}.{}'.format(section,subsection))
             #print(topic)
             files = files + topic.split('$SOF')
             subsection += 1
             #if subsection == 2:
             #   break
  files = list(filter(None,files))
  #print(files)
  fln = 1
  if split:
   for fl in files:
     #print(fl.split('\n')[0])
     filename = fl.split('\n')[0]
     filename = filename.replace('.', '_')
     filename = filename.replace('#', '')
     filename = filename.strip()
     name_parts = filename.split(' ', 1)
     #Below part switches the numbering and title
     #if len(name_parts) > 1:
     #  filename = name_parts[1] + '_' + name_parts[0]
     #else:
     #  filename = name_parts[0]
     filename = filename.replace(' ', '_')
     filename = filename.replace('/', '_tai_')
     filename = filename.replace('?', '')
     #name_parts = 
     #filename = name_parts[1] + '_' + name_parts[0]
     #print(filename)
     f = open('output/{}.md'.format(filename),'w', encoding='utf-8')
     f.write(fl)
     f.close()
     fln += 1
  else:
     filename = 'etuusohje_koko'
     f = open('output/{}.md'.format(filename),'w', encoding='utf-8')
     partnames = [str]
     for fl in files:
        #split = re.split(r'\n', fl)
        #print('head', split[1])
        #name = split[1]
        #name = re.sub(r'(#)', '', name)
        #name = re.sub(r'\/', '_tai_', name)
        #name = name.strip()
        #partnames.append(name)
        #f.writelines(split[2:])
        f.write(fl)
     f.close()
     if useLangchain:
        text_splitter = RecursiveCharacterTextSplitter(
           chunk_size = 2048,
           chunk_overlap = 150,
           #length_function = lambda x: num_tokens_from_string(x),
           length_function = len,
           add_start_index = True,
        )
        headers_to_split_on = [
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3'),
        ('####', 'Header 4'),
        ('#####', 'Header 5'),
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        f = open('output/{}.md'.format(filename),'r', encoding='utf-8')
        texts = text_splitter.split_text(f.read())
        #texts = text_splitter.create_documents([f.read()])
        f.close()
        if(os.path.exists('output_lc')):
            shutil.rmtree('output_lc')
        #os.removedirs('output_lc')
        os.makedirs('output_lc', exist_ok=True)
        lc_idx = 0
        print('length of texts', len(texts))
        for t in texts:
          f = open('output_lc/{}.json'.format(lc_idx),'w', encoding='utf-8')
          f.write(t.json() + '\n')
          f.close()
          lc_idx += 1

def get_topic(node: ET.Element, section, subsection=-1, optionalsubsection=-1):
   depth = node.attrib['otherprops']
   if section == '1.1':
      print('section:', section, subsection, optionalsubsection, depth)
   #print('depth', depth)
   x = 1
   y = 1
   text = ''
   #if depth != 'depth2':
   text += '$SOF'
   for child in node:
      if child.tag == 'title':
         if subsection>0:
            if depth == 'depth2':
              section = '{}.{}'.format(section, alphabets[optionalsubsection])
              #9.10.2023 adding letters makes the numbering skip
              #x -= 1
            else:
              section = '{}.{}'.format(section, subsection)
         depth = len(section.split('.'))
         section_title = '\n'+'#'*depth + ' {} {}'.format(section, get_text(child))
         text += section_title
      if child.tag == 'body':
        for elem in child:
            if elem.tag == 'p':
                txt = get_text(elem)
                text += '\n' + txt
            if elem.tag == 'ul':
                txt = get_list(elem)
                text += '\n' + txt
            if elem.tag == 'ol':
                txt = get_list(elem, True)
                text += '\n' + txt
            if elem.tag == 'example':
                txt = get_text(elem)
                text += '\n>ESIMERKKI: ' + txt + '\n'
                #print(txt)
        #if depth != 'depth2':
        #  text += '$EOF'
      if child.tag == 'topic':
         #if section == '1.3' and x == 1:
            #print(section, x, y, depth)
         topic = get_topic(child, section, x, y)
         #print(topic.split(' ')[1][-1])
         if topic.split(' ')[1][-1].isalpha():
            #print('ends with alphabet')
            x -= 1
         text = text + '\n'+ topic
         x += 1
         y += 1
   return text
         
def get_list(node: ET.Element, ordered=False, prev_index=-1):
   ''' Parse a list element '''
   text = ''
   n = 1
   for child in node:
      #print(child.tag, child.attrib)
      if child.tag == 'li':
         if ordered:
            text += ' {}. '.format(n) + get_text(child)+'\n'
            n += 1
         else:
            text += ' - ' + get_text(child)+'\n'
         for nested in child:
            #print(nested.tag, nested.attrib)
            if nested.tag == 'ul':
              get_list(nested)
            if nested.tag == 'ol':
              get_list(nested)
   return text

def get_text(node: ET.Element):
    '''Gets text out of an XML Node'''

    # Get initial text
    text = node.text if node.text else ''
    text = ' '.join(text.split())
    # Get all text from child nodes recursively
    for child_node in node:
        if len(text) > 0:
          text += ' ' + get_text(child_node)
        else:
          text += get_text(child_node)
    # Get text that occurs after child nodes
    #text += node.tail if node.tail else ''
    text += ' ' + ' '.join(node.tail.split()) if node.tail else ''
    return text

if __name__ == '__main__':
   main(sys.argv[1:])