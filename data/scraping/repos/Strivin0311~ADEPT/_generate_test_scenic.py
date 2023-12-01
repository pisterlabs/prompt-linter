import os
import os.path
from pathlib import Path
from urllib import response
from click import command
import openai
import time
import random

# openai.organization = "org-FboVbR5bULjjVglwPOQ71SCN"
openai.organization = "org-xPHZpZxKEUaFG5YQZ0twHK1h"
openai.api_key = os.getenv("OPENAI_API_KEY")

def getanswer(R, Q):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Q: "+R +" "+Q+"\nA:",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    time.sleep(5)
    return response["choices"][0]["text"]

def get_accident_place(R):
    a1 = getanswer(R,"Did the accident take place at an intersection?")
    if "no" in a1.lower():
        return "no"
    a2 = getanswer(R,"Did the accident take place at an 4-way intersection?")
    if "no" in a1.lower():
        return "3way"
    return "4way"

def YorN_QA(R, Q):
    a = getanswer(R,Q)
    if "no" in a.lower():
        return "no"
    elif "yes" in a.lower():
        return "yes"
    else:
        return "unknown"

def get_lane_relathionship(R):
    a = getanswer(R,"Is vehicle 2 in the left or right lane of vehicle 1, or neither?")
    if "left" in a.lower():
        return "left"
    elif "right" in a.lower():
        return "right"
    else:
        return "mid"

def Distance_QA(R, Q):
    a = getanswer(R,Q+" Answer with numbers.")
    try:
        a = float(a)
    except ValueError as e:
        pass
    finally:
        a = -1
    return a

def get_vehicle_type(R):
    vehicletype = {
        "lorry":["vehicle.carlamotors.carlacola"],
        "pickup":["vehicle.tesla.cybertruck"],
        "suv":["vehicle.nissan.patrol", "vehicle.audi.etron"],
        "jeep":["vehicle.jeep.wrangler_rubicon"],
        "compact car":["vehicle.audi.a2", "vehicle.nissan.micra", "vehicle.bmw.isetta", "vehicle.citroen.c3", "vehicle.seat.leon"],
        "sedan":["vehicle.chevrolet.impala", "vehicle.bmw.grandtourer", "vehicle.mini.cooperst", "vehicle.toyota.prius", "vehicle.tesla.model3", "vehicle.lincoln.mkz2017", "vehicle.lincoln2020.mkz2020", "vehicle.charger2020.charger2020"],
        "coupe":["vehicle.mercedesccc.mercedesccc", "vehicle.audi.tt", "vehicle.mercedes-benz.coupe", "vehicle.mustang.mustang"],
        "motorcycle":["vehicle.harley-davidson.low_rider", "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja"],
        "small bus":["vehicle.volkswagen.t2"],
        "police car":["vehicle.chargercop2020.chargercop2020", "vehicle.dodge_charger.police"],
        "bike":["vehicle.bh.crossbike", "vehicle.gazelle.omafiets", "vehicle.diamondback.century"]
    }
    vehicletypelist = list(vehicletype.keys())
    vehicletypelist[-1] = "or "+ vehicletypelist[-1]
    vehicletypestr = ", ".join(vehicletypelist)
    v1type_qa = getanswer(R,"Is Vehicle 1 "+vehicletypestr+"?")
    v2type_qa = getanswer(R,"Is Vehicle 2 "+vehicletypestr+"?")
    v1type = v2type = "sedan"
    for t in vehicletype.keys():
        if t.lower() in v1type_qa.lower():
            v1type = t
        if t.lower() in v2type_qa.lower():
            v2type = t
    v1model = random.choice(vehicletype[v1type])
    v2model = random.choice(vehicletype[v2type])
    return v1model, v2model

def get_command(R, Q):
    a = getanswer(R, Q)
    if "left" in a.lower():
        return "LEFT"
    elif "right" in a.lower():
        return "RIGHT"
    else:
        return "STRAIGHT"

def write_scenic(text, R, command):
    f = open("test_scenic/CNT.txt","r")
    cnt = f.read()
    f.close()
    cnt = int(cnt)
    cnt += 1
    cnt = str(cnt)
    f=open("test_scenic/CNT.txt",'w')
    f.write(cnt)
    f.close()
    i = cnt
    if "TOWN" in text:
        for t in ["Town01","Town02","Town03","Town04","Town05","Town06","Town07"]:
            tmp_text = text.replace("TOWN", t)
            path = "test_scenic/"+t+"/"+str(i) +"_"+ command +".scenic"
            tmp_text = tmp_text + "# " + R
            f=open(path,'w',encoding="utf-8")
            f.write(tmp_text)
            f.close()
            print("path:", path)
    else:
        path = "test_scenic/Town02/"+str(i) +"_"+ command +".scenic"
        text = text + "# " + R
        f=open(path,'w',encoding="utf-8")
        f.write(text)
        f.close()
        print("path:", path)


# while(True):
if __name__ == "__main__":
    report = input("Input crash report:")
# def func(report):
    v1model, v2model = get_vehicle_type(report)
    # print(v1model,v2model)
    intersection = get_accident_place(report)
    # print("intersection:", intersection)
    if intersection == "no":
        #直路
        same_direction = YorN_QA(report,"Are Vehicle 1 and Vehicle 2 driving in the same direction?")
        # print("Same_direction:",same_direction)
        if same_direction == "yes" or same_direction == "unknown":
            #同向行驶
            f = open("scenic_template/template_1_non_same.scenic","r")
            text = f.read()
            f.close()
            text = text.replace("V1_MODEL","\""+v1model+"\"")
            text = text.replace("V2_MODEL","\""+v2model+"\"")

            v1_v2_dist_f = Distance_QA(report,"How many meters is vehicle 2 in front of vehicle 1 at the beginning?")
            v1_v2_dist_f = int(v1_v2_dist_f)
            v1_v2_dist_f = max(v1_v2_dist_f, 10)
            v1_v2_dist_f = min(v1_v2_dist_f, 50)
            text = text.replace("DISTF", str(v1_v2_dist_f))

            lane = get_lane_relathionship(report)
            # print("lane:",lane)
            if lane == "right":
                #同向-v1-v2
                v1_v2_dist_r = Distance_QA(report,"How many meters is vehicle 2 in right of vehicle 1 at the beginning?")
                v1_v2_dist_r = int(v1_v2_dist_r)
                v1_v2_dist_r = max(v1_v2_dist_r, 2)
                v1_v2_dist_r = min(v1_v2_dist_r, 10)
                text = text.replace("V2_DIRE", "right")
                text = text.replace("V2_DEG", "30")
                text = text.replace("DISTLR", str(v1_v2_dist_r))

            elif lane == "left":
                #同向-v2-v1
                v1_v2_dist_l = Distance_QA(report,"How many meters is vehicle 2 in left of vehicle 1 at the beginning?")
                v1_v2_dist_l = int(v1_v2_dist_l)
                v1_v2_dist_l = max(v1_v2_dist_l, 2)
                v1_v2_dist_l = min(v1_v2_dist_l, 10)
                text = text.replace("V2_DIRE", "left")
                text = text.replace("V2_DEG", "-30")
                text = text.replace("DISTLR", str(v1_v2_dist_l))
            else:
                text = text.replace("V2_DIRE", "left")
                text = text.replace("V2_DEG", "0")
                text = text.replace("DISTLR", 0)
            
            write_scenic(text,report,"FOLLOW_LANE")
        else:
            #异向行驶
            f = open("scenic_template/template_2_non_diff.scenic","r")
            text = f.read()
            f.close()
            text = text.replace("V1_MODEL","\""+v1model+"\"")
            text = text.replace("V2_MODEL","\""+v2model+"\"")

            v1_v2_dist_f = Distance_QA(report,"How many meters is vehicle 2 in front of vehicle 1 at the beginning?")
            v1_v2_dist_f = int(v1_v2_dist_f)
            v1_v2_dist_f = max(v1_v2_dist_f, 20)
            v1_v2_dist_f = min(v1_v2_dist_f, 50)
            text = text.replace("DISTF", str(v1_v2_dist_f))

            write_scenic(text,report,"FOLLOW_LANE")
    elif intersection == "4way":
        #十字路口
        f = open("scenic_template/template_3_4way.scenic","r")
        text = f.read()
        f.close()
        text = text.replace("V1_MODEL","\""+v1model+"\"")
        text = text.replace("V2_MODEL","\""+v2model+"\"")

        c1 = get_command(report, "Vehicle 1 is going straight, turning left or right?")
        c2 = get_command(report, "Vehicle 2 is going straight, turning left or right?")

        if c1 == "LEFT":
            text = text.replace("V1_COMMAND","LEFT_TURN")
        elif c1 == "RIGHT":
            text = text.replace("V1_COMMAND","RIGHT_TURN")
        else:
            text = text.replace("V1_COMMAND","STRAIGHT")
        if c2 == "LEFT":
            text = text.replace("V2_COMMAND","LEFT_TURN")
        elif c2 == "RIGHT":
            text = text.replace("V2_COMMAND","RIGHT_TURN")
        else:
            text = text.replace("V2_COMMAND","STRAIGHT")
        
        write_scenic(text,report, c1)

    elif intersection == "3way":
        #丁字路口
        f = open("scenic_template/template_4_3way.scenic","r")
        text = f.read()
        f.close()
        text = text.replace("V1_MODEL","\""+v1model+"\"")
        text = text.replace("V2_MODEL","\""+v2model+"\"")

        c1 = get_command(report, "Vehicle 1 is going straight, turning left or right?")
        c2 = get_command(report, "Vehicle 2 is going straight, turning left or right?")

        if c1 == "LEFT":
            text = text.replace("V1_COMMAND","LEFT_TURN")
        elif c1 == "RIGHT":
            text = text.replace("V1_COMMAND","RIGHT_TURN")
        else:
            text = text.replace("V1_COMMAND","STRAIGHT")
        if c2 == "LEFT":
            text = text.replace("V2_COMMAND","LEFT_TURN")
        elif c2 == "RIGHT":
            text = text.replace("V2_COMMAND","RIGHT_TURN")
        else:
            text = text.replace("V2_COMMAND","STRAIGHT")
        
        write_scenic(text,report, c1)
    print("DONE")