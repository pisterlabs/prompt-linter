import os
import openai
import ast

three = """
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   5 125 172 249 254 255 163  24   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  54 205 244 162 154 102 190 253  65   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0  59 237 118  30   0   0   0  30 253  65   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0 105 175   0   0   0   0   0  19 253  65   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   7  25   0   0   0   0   0 156 232  34   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  16 241 113   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  17 173 209  10   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0 169 232  35   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  50 198 192 203 252 248 235 155  88   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  70 253 234 124  90  90  93 171 231 235
   58   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0  27   0   0   0   0   0   0  35 183
  240  30   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  10
  218 183   3   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  135 253  32   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   52 253  32   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  162 253  32   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   7
  214 211  16   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 149
  253  51   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0 159  85   0   0   0   0   0  31 208 248
  173   4   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0 130 245 192 155 155 208 238 243 198 106
    3   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   2  50 161 245 229 171 142  58   2   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
"""

eight = """
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 123 253 200  78  25   0   0
    0  29 130  36   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  80 253 252 252 252 226  71   0
    8 197 252 243  83   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 115 253 252 136 202 252 211   0
   84 252 252 232  40   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 211 253 252  21  21 198 211  22
  237 252 252 140   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 211 253 217  12   0  18  53 199
  252 252 164  18   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 194 254 253  21   0   0  61 253
  253 243  79   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 106 253 252  21   0  36 227 252
  252 110   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  97 253 252  29   6 162 253 252
  215   7   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 227 252 196 134 252 253 231
   16   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 104 252 249 239 252 236  54
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  43 253 253 253 253 132   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  11 205 252 252 217   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  48 247 252 252 147   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  27 218 252 252 252 147   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 150 253 252 252 252 147   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0  55 236 255 239 206 253 147   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0 169 252 253 152 239 252 147   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  20 246 252 253 252 252 247  91   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  22 252 252 253 252 221  98   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   4 182 252 191 112  21   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
"""

MODEL = "gpt-3.5-turbo" 
# MODEL = "text-davinci-003"

def davinci(array):
    prompt = "What possible values could it represent? \n{}".format(array)

    completion = openai.ChatCompletion.create(
        model=MODEL, 
        messages = [
         {"role": "system", "content": "You are an AI that evaluates the 28 by 28 matrix outputs what number the image represents and also what it can be confused with."},
         {"role": "system", "content" : prompt},
         # {"role": "assistant", "content": "This is a handwritten digit image from the MNIST dataset. To determine the value it represents, we can feed it into a machine learning model trained on the MNIST dataset. However, without doing so, we can make an educated guess based on the image."},
         # {"role": "system", "content" : "Make an educated guess. Return nothing but a python array, such as: [2,5,9]."}
       ]
    )
    out = completion.choices[0].message.content
    return ast.literal_eval(out)


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    out = davinci(three)
    print(out)


if __name__ == "__main__":
  main()