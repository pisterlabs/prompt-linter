
import pandas as pd
import pickle
import os
import openai

df = pd.read_csv("../../data/avr/test.csv")
functions = df["source"].tolist()
functions_cut = []

for func in functions:
    # word-level 400 tokens would equal to more than 512 subword tokens
    f_cut = func.split(" ")[:400]
    f_cut = " ".join(f_cut)
    functions_cut.append(f_cut)

print("total number of samples: ", len(functions_cut))

openai.api_key = "YOUR-KEY-HERE"

start = 0
for idx in range(start, len(functions_cut)):
    one_function = functions_cut[idx]
    messages = [{"role": "user",
                 "content": 
"""Example Vulnerable Code 1:
<S2SV_StartBug> static INLINE void add_token ( TOKENEXTRA * * t , const vp9_prob * context_tree , <S2SV_EndBug> <S2SV_StartBug> int16_t extra , uint8_t token , <S2SV_EndBug> uint8_t skip_eob_node 
, unsigned int * counts ) { ( * t ) -> token = token ; ( * t ) -> extra = extra ; ( * t ) -> context_tree = context_tree ; ( * t ) -> skip_eob_node = skip_eob_node ; ( * t ) ++ ; ++ counts [ token ] ; }

Example Repair Patch 1:
<S2SV_ModStart> t , const vpx_prob <S2SV_ModEnd> * context_tree , <S2SV_ModStart> * context_tree , int32_t <S2SV_ModEnd> extra , uint8_t

Example Vulnerable Code 2:
void cisco_autorp_print ( netdissect_options * ndo , register const u_char * bp , register u_int len ) { int type ; int numrps ; int hold ; <S2SV_StartBug> ND_TCHECK ( bp [ 0 ] ) ; <S2SV_EndBug> ND_PRINT ( ( ndo , "<S2SV_blank>auto-rp<S2SV_blank>" ) ) ; type = bp [ 0 ] ; switch ( type ) { case 0x11 : ND_PRINT ( ( ndo , "candidate-advert" ) ) ; break ; case 0x12 : ND_PRINT ( ( ndo , "mapping" ) ) ; break ; default : ND_PRINT ( ( ndo , "type-0x%02x" , type ) ) ; break ; } ND_TCHECK ( bp [ 1 ] ) ; numrps = bp [ 1 ] ; ND_TCHECK2 ( bp [ 2 ] , 2 ) ; ND_PRINT ( ( ndo , "<S2SV_blank>Hold<S2SV_blank>" ) ) ; hold = EXTRACT_16BITS ( & bp [ 2 ] ) ; if ( hold ) unsigned_relts_print ( ndo , EXTRACT_16BITS ( & bp [ 2 ] ) ) ; else ND_PRINT ( ( ndo , "FOREVER" ) ) ; bp += 8 ; len -= 8 ; while ( numrps -- ) { int nentries ; char s ; <S2SV_StartBug> ND_TCHECK2 ( bp [ 0 ] , 4 ) ; <S2SV_EndBug> ND_PRINT ( ( ndo , "<S2SV_blank>RP<S2SV_blank>%s" , ipaddr_string ( ndo , bp ) ) ) ; <S2SV_StartBug> ND_TCHECK ( bp [ 4 ] ) ; <S2SV_EndBug> <S2SV_StartBug> switch ( bp [ 4 ] & 0x3 ) { <S2SV_EndBug> case 0 : ND_PRINT ( ( ndo , "<S2SV_blank>PIMv?" ) ) ; break ; case 1 : ND_PRINT ( ( ndo , "<S2SV_blank>PIMv1" ) ) ; break ; case 2 : ND_PRINT ( ( ndo , "<S2SV_blank>PIMv2" ) ) ; break ; case 3 : ND_PRINT ( ( ndo , "<S2SV_blank>PIMv1+2" ) ) ; break ; } <S2SV_StartBug> if ( bp [ 4 ] & 0xfc ) <S2SV_EndBug> <S2SV_StartBug> ND_PRINT ( ( ndo , "<S2SV_blank>[rsvd=0x%02x]" , bp [ 4 ] & 0xfc ) ) ; <S2SV_EndBug> <S2SV_StartBug> ND_TCHECK ( bp [ 5 ] ) ; <S2SV_EndBug> <S2SV_StartBug> nentries = bp [ 5 ] ; <S2SV_EndBug> <S2SV_StartBug> bp += 6 ; len -= 6 ; <S2SV_EndBug> s = '<S2SV_blank>' ; for ( ; nentries ; nentries -- ) { <S2SV_StartBug> ND_TCHECK2 ( bp [ 0 ] , 6 ) ; <S2SV_EndBug> ND_PRINT ( ( ndo , "%c%s%s/%d" , s , bp [ 0 ] & 1 ? "!" : "" , ipaddr_string ( ndo , & bp [ 2 ] ) , bp [ 1 ] ) ) ; if ( bp [ 0 ] & 0x02 ) { ND_PRINT ( ( ndo , "<S2SV_blank>bidir" ) ) ; } if ( bp 
[ 0 ] & 0xfc ) { ND_PRINT ( ( ndo , "[rsvd=0x%02x]" , bp [ 0 ] & 0xfc ) ) ; } s = ',' ; bp += 6 ; len -= 6 ; } } return ; trunc : ND_PRINT ( ( ndo , "[|autorp]" ) ) ; return ; }

Example Repair Patch 2:
<S2SV_ModStart> int hold ; if ( len < 8 ) goto trunc ; <S2SV_ModStart> char s ; if ( len < 4 ) goto trunc ; <S2SV_ModStart> ) ) ; bp += 4 ; len -= 4 ; if ( len < 1 ) goto trunc ; <S2SV_ModStart> ( bp [ 0 <S2SV_ModEnd> ] ) ; <S2SV_ModStart> ( bp [ 0 <S2SV_ModEnd> ] & 0x3 <S2SV_ModStart> ( bp [ 0 <S2SV_ModEnd> ] & 0xfc <S2SV_ModStart> , bp [ 0 <S2SV_ModEnd> ] & 0xfc <S2SV_ModStart> ) ) ; bp 
+= 1 ; len -= 1 ; if ( len < 1 ) goto trunc ; <S2SV_ModStart> ( bp [ 0 <S2SV_ModEnd> ] ) ; <S2SV_ModStart> = bp [ 0 <S2SV_ModEnd> ] ; bp <S2SV_ModStart> ; bp += 1 ; len -= 1 <S2SV_ModEnd> ; s = <S2SV_ModStart> -- ) { if ( len < 6 ) goto trunc ;

Example Vulnerable Code 3:
int add_control_packet ( struct mt_packet * packet , enum mt_cptype cptype , void * cpdata , unsigned short data_len ) { unsigned char * data = packet -> data + packet -> size ; unsigned int act_size = data_len + ( cptype == MT_CPTYPE_PLAINDATA ? 0 : MT_CPHEADER_LEN ) ; <S2SV_StartBug> if ( packet -> size + act_size > MT_PACKET_LEN ) { <S2SV_EndBug> fprintf ( stderr , _ ( "add_control_packet:<S2SV_blank>ERROR,<S2SV_blank>too<S2SV_blank>large<S2SV_blank>packet.<S2SV_blank>Exceeds<S2SV_blank>%d<S2SV_blank>bytes\\n" ) , MT_PACKET_LEN ) ; return - 1 ; } if ( cptype == MT_CPTYPE_PLAINDATA ) { memcpy ( data , cpdata , data_len ) ; packet -> size += data_len ; return data_len ; } memcpy ( data , mt_mactelnet_cpmagic , sizeof ( mt_mactelnet_cpmagic ) ) ; data [ 4 ] = cptype ; # if 
BYTE_ORDER == LITTLE_ENDIAN { unsigned int templen ; templen = htonl ( data_len ) ; memcpy ( data + 5 , & templen , sizeof ( templen ) ) ; } # else memcpy ( data + 5 , & data_len , sizeof ( data_len ) ) ; # endif if ( data_len > 0 ) { memcpy ( data + MT_CPHEADER_LEN , cpdata , data_len ) ; } packet -> size += act_size ; return act_size ; }

Example Repair Patch 3:
<S2SV_ModStart> ; if ( <S2SV_ModEnd> act_size > MT_PACKET_LEN <S2SV_ModStart> act_size > MT_PACKET_LEN - packet -> size"""},
                {"role": "user", "content": "Generate repair tokens for vulnerable tokens below:"},
                {"role": "user", "content": one_function},
                {"role": "user", "content": "Do not generate the whole function. Do not generate any explanation. The return format should strictly follow the Example Repair Patch I provided."}]
    print("ChatGPT start...")
    
    model_name = "gpt-3.5-turbo"

    # create a chat completion
    try:
        chat_completion = openai.ChatCompletion.create(model=model_name, messages=messages)
    except:
        print("ChatGPT server error!")
        print(f"terminated at index {idx} ...")
        print("try to restart...")
        chat_completion = openai.ChatCompletion.create(model=model_name, messages=messages)
    print("ChatGPT completed...")
    with open(f"../response/avr_files/{model_name}/{idx}.pkl", "wb+") as f:
        pickle.dump(chat_completion, f)
        
    if '<S2SV_ModStart>' not in chat_completion.choices[0].message.content:
        print("error no <S2SV_ModStart> returned", "\n", chat_completion.choices[0].message.content)
        print(f"terminated at index {idx} ...")
        exit()
    else:
        print(chat_completion.choices[0].message.content)
print("done")