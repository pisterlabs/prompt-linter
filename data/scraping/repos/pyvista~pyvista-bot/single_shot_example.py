"""
Example: Generate a pyvista source function using single shot training.

"""
import inspect
import openai
from get_funcs import get_funcs, get_body
from vtk_info import get_cls_info
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

 
from key import API_KEY

openai.api_key = API_KEY

# PyVista test located at {test_func.__module__}.{test_func.__qualname__}
# {pv_test_source}

def gen_src(pv_func, test_func, vtk_info_in, vtk_info_out):
    pv_source = get_body(pv_func)
    # pv_test_source = inspect.getsource(test_func)

    gpt_prompt = f"""
Generate the source code of a single Python function given this input, output pattern.

Input:
{vtk_info_in['cls_name']}
{vtk_info_in['fnames']}

Output:
# PyVista function located at: {pv_func.__qualname__}
{pv_source}

Input:
{vtk_info_out['cls_name']}
{vtk_info_out['fnames']}

Output:
"""
    # tok = tokenizer(gpt_prompt)
    # if len(tok['input_ids']) > 2048:
    #     raise ValueError('Input prompt too large')

    src_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=gpt_prompt,
        temperature=0.5,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop='\n\n',
        stop='Input:',
    )

    return src_response['choices'][0]['text']


vtk_cls = 'vtkAppendPolyData'
print('Acquiring PyVista source...')
pv_funcs, pv_func, test_funcs, test_func = get_funcs(vtk_cls)

print('Acquiring VTK class definitions...')
vtk_cls_info = get_cls_info(vtk_cls)
vtk_info_out = get_cls_info('vtkEllipseArcSource')

print('Querying OpenAI...')
out = gen_src(pv_func, test_func, vtk_cls_info, vtk_info_out)
print(out)
