from generate_function import create_image_album_llm, album, llm_create_story, load_diffuser_model, load_llm_model, clear_cuda_memory
import streamlit as st
import zipfile
import io
import torch
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"


def main():
    cache = torch.cuda.memory_cached() / 1024 ** 3
    print(cache)
    if cache >= 4.4 :
        clear_cuda_memory()
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"CUDA Memory Cached: {torch.cuda.memory_cached() / 1024 ** 3:.2f} GB")
    device = "cuda"
 
    st.subheader("Please enter in your prompt/suggestion:")
    input = st.text_area("")
    if st.button("Create AI-generated story"):
        llm = load_llm_model(device, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
        story = llm_create_story(llm=llm, suggestion=input)
        st.write(story)
        st.write("Try pressing the generate button again if no images are generated")
        pipe = load_diffuser_model(device)
        create_image_album_llm(input_string=story, pipeline=pipe)
        
  
        
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for i, img in enumerate(album):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                img_bytes = img_buffer.getvalue()
                zipf.writestr(f"scene_{i + 1}.jpg", img_bytes)

    zip_buffer.seek(0)
    st.download_button(
        label="Download",
        data=zip_buffer,
        file_name="generated_scenes.zip",
        key="download_all_button",
    )


            
    # Using object notation

    # Using "with" notation
    with st.sidebar:
        st.title('FAQ')
        st.subheader('Github : https://github.com/LPK99/AI-Storyboard-Stable-Diffusion-')
        st.write('Developed by Duc Luu')


if __name__ == "__main__":
    main()
    