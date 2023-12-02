import random
import textwrap
import openai
from datetime import datetime
from decouple import config
# DEV NOTES
# (ASCII INTERFACES TO ADD IN ORDER OF MVP BOT USERFLOW) 
# rites, glossolalia, bazzar, monolith, oppenheimer, quantum_analysis, wormhole 
# (DESCRIPTIONS) 
# rites = ascii interface for first time onboarding account creation
# glossolalia = ascii interface for Login page with unique voice glossolalia detection login 
# monolith = ascii interface for main bot page 
# oppenheimer = abort all trades ascii interface
# quantum_analysis = interface for find co integrated pairs
# divination = ascii interface for manage exits
# wormhole = ascii interface for opening trades

i_ching_names = [
    {"No": "01", "Gua": "䷀", "TC": "乾", "Pinyin": "qián", "SC": "乾", "English": "Initiating", "Binary": "111111", "OD": "02"},
    {"No": "02", "Gua": "䷁", "TC": "坤", "Pinyin": "kūn", "SC": "坤", "English": "Responding", "Binary": "000000", "OD": "01"},
    {"No": "03", "Gua": "䷂", "TC": "屯", "Pinyin": "zhūn", "SC": "屯", "English": "Beginning", "Binary": "010001", "OD": "50"},
    {"No": "04", "Gua": "䷃", "TC": "蒙", "Pinyin": "méng", "SC": "蒙", "English": "Childhood", "Binary": "100010", "OD": "49"},
    {"No": "05", "Gua": "䷄", "TC": "需", "Pinyin": "xū", "SC": "需", "English": "Needing", "Binary": "010111", "OD": "35"},
    {"No": "06", "Gua": "䷅", "TC": "訟", "Pinyin": "sòng", "SC": "讼", "English": "Contention", "Binary": "111010", "OD": "36"},
    {"No": "07", "Gua": "䷆", "TC": "師", "Pinyin": "shī", "SC": "师", "English": "Multitude", "Binary": "000010", "OD": "13"},
    {"No": "08", "Gua": "䷇", "TC": "比", "Pinyin": "bǐ", "SC": "比", "English": "Union", "Binary": "010000", "OD": "14"},
    {"No": "09", "Gua": "䷈", "TC": "小畜", "Pinyin": "xiǎo chù", "SC": "小畜", "English": "Little Accumulation", "Binary": "110111", "OD": "16"},
    {"No": "10", "Gua": "䷉", "TC": "履", "Pinyin": "lǚ", "SC": "履", "English": "Fulfillment", "Binary": "111011", "OD": "15"},
    {"No": "11", "Gua": "䷊", "TC": "泰", "Pinyin": "tài", "SC": "泰", "English": "Advance", "Binary": "000111", "OD": "12"},
    {"No": "12", "Gua": "䷋", "TC": "否", "Pinyin": "pǐ", "SC": "否", "English": "Hindrance", "Binary": "111000", "OD": "11"},
    {"No": "13", "Gua": "䷌", "TC": "同人", "Pinyin": "tóng rén", "SC": "同人", "English": "Seeking Harmony", "Binary": "111101", "OD": "07"},
    {"No": "14", "Gua": "䷍", "TC": "大有", "Pinyin": "dà yǒu", "SC": "大有", "English": "Great Harvest", "Binary": "101111", "OD": "08"},
    {"No": "15", "Gua": "䷎", "TC": "謙", "Pinyin": "qiān", "SC": "谦", "English": "Humbleness", "Binary": "000100", "OD": "10"},
    {"No": "16", "Gua": "䷏", "TC": "豫", "Pinyin": "yù", "SC": "豫", "English": "Delight", "Binary": "001000", "OD": "09"},
    {"No": "17", "Gua": "䷐", "TC": "隨", "Pinyin": "suí", "SC": "随", "English": "Following", "Binary": "011001", "OD": "18"},
    {"No": "18", "Gua": "䷑", "TC": "蠱", "Pinyin": "gǔ", "SC": "蛊", "English": "Remedying", "Binary": "100110", "OD": "17"},
    {"No": "19", "Gua": "䷒", "TC": "臨", "Pinyin": "lín", "SC": "临", "English": "Approaching", "Binary": "000011", "OD": "33"},
    {"No": "20", "Gua": "䷓", "TC": "觀", "Pinyin": "guān", "SC": "观", "English": "Watching", "Binary": "110000", "OD": "34"},
    {"No": "21", "Gua": "䷔", "TC": "噬嗑", "Pinyin": "shì kè", "SC": "噬嗑", "English": "Eradicating", "Binary": "101001", "OD": "48"},
    {"No": "22", "Gua": "䷕", "TC": "賁", "Pinyin": "bì", "SC": "贲", "English": "Adorning", "Binary": "100101", "OD": "47"},
    {"No": "23", "Gua": "䷖", "TC": "剝", "Pinyin": "bō", "SC": "剥", "English": "Falling Away", "Binary": "100000", "OD": "43"},
    {"No": "24", "Gua": "䷗", "TC": "復", "Pinyin": "fù", "SC": "复", "English": "Turning Back", "Binary": "000001", "OD": "44"},
    {"No": "25", "Gua": "䷘", "TC": "無妄", "Pinyin": "wú wàng", "SC": "无妄", "English": "Without Falsehood", "Binary": "111001", "OD": "46"},
    {"No": "26", "Gua": "䷙", "TC": "大畜", "Pinyin": "dà chù", "SC": "大畜", "English": "Great Accumulation", "Binary": "100111", "OD": "45"},
    {"No": "27", "Gua": "䷚", "TC": "頤", "Pinyin": "yí", "SC": "颐", "English": "Nourishing", "Binary": "100001", "OD": "28"},
    {"No": "28", "Gua": "䷛", "TC": "大過", "Pinyin": "dà guò", "SC": "大过", "English": "Great Exceeding", "Binary": "011110", "OD": "27"},
    {"No": "29", "Gua": "䷜", "TC": "坎", "Pinyin": "kǎn", "SC": "坎", "English": "Darkness", "Binary": "010010", "OD": "30"},
    {"No": "30", "Gua": "䷝", "TC": "離", "Pinyin": "lí", "SC": "离", "English": "Brightness", "Binary": "101101", "OD": "29"},
    {"No": "31", "Gua": "䷞", "TC": "咸", "Pinyin": "xián", "SC": "咸", "English": "Mutual Influence", "Binary": "011100", "OD": "41"},
    {"No": "32", "Gua": "䷟", "TC": "恆", "Pinyin": "héng", "SC": "恒", "English": "Long Lasting", "Binary": "001110", "OD": "42"},
    {"No": "33", "Gua": "䷠", "TC": "遯", "Pinyin": "dùn", "SC": "遯", "English": "Retreat", "Binary": "111100", "OD": "19"},
    {"No": "34", "Gua": "䷡", "TC": "大壯", "Pinyin": "dà zhuàng", "SC": "大壮", "English": "Great Strength", "Binary": "001111", "OD": "20"},
    {"No": "35", "Gua": "䷢", "TC": "晉", "Pinyin": "jìn", "SC": "晋", "English": "Proceeding Forward", "Binary": "101000", "OD": "05"},
    {"No": "36", "Gua": "䷣", "TC": "明夷", "Pinyin": "míng yí", "SC": "明夷", "English": "Brilliance Injured", "Binary": "101000", "OD": "05"},
    {"No": "37", "Gua": "䷤", "TC": "家人", "Pinyin": "jiā rén", "SC": "家人", "English": "Household", "Binary": "110101", "OD": "40"},
    {"No": "38", "Gua": "䷥", "TC": "睽", "Pinyin": "kuí", "SC": "睽", "English": "Diversity", "Binary": "101011", "OD": "39"},
    {"No": "39", "Gua": "䷦", "TC": "蹇", "Pinyin": "jiǎn", "SC": "蹇", "English": "Hardship", "Binary": "010100", "OD": "38"},
    {"No": "40", "Gua": "䷧", "TC": "解", "Pinyin": "xiè", "SC": "解", "English": "Relief", "Binary": "001010", "OD": "37"},
    {"No": "41", "Gua": "䷨", "TC": "損", "Pinyin": "sǔn", "SC": "损", "English": "Decreasing", "Binary": "100011", "OD": "31"},
    {"No": "42", "Gua": "䷩", "TC": "益", "Pinyin": "yì", "SC": "益", "English": "Increasing", "Binary": "110001", "OD": "32"},
    {"No": "43", "Gua": "䷪", "TC": "夬", "Pinyin": "guài", "SC": "夬", "English": "Eliminating", "Binary": "011111", "OD": "23"},
    {"No": "44", "Gua": "䷫", "TC": "姤", "Pinyin": "gòu", "SC": "姤", "English": "Encountering", "Binary": "111110", "OD": "24"},
    {"No": "45", "Gua": "䷬", "TC": "萃", "Pinyin": "cuì", "SC": "萃", "English": "Bringing Together", "Binary": "011000", "OD": "26"},
    {"No": "46", "Gua": "䷭", "TC": "升", "Pinyin": "shēng", "SC": "升", "English": "Growing Upward", "Binary": "000110", "OD": "25"},
    {"No": "47", "Gua": "䷮", "TC": "困", "Pinyin": "kùn", "SC": "困", "English": "Exhausting", "Binary": "011010", "OD": "22"},
    {"No": "48", "Gua": "䷯", "TC": "井", "Pinyin": "jǐng", "SC": "井", "English": "Replenishing", "Binary": "010110", "OD": "21"},
    {"No": "49", "Gua": "䷰", "TC": "革", "Pinyin": "gé", "SC": "革", "English": "Abolishing The Old", "Binary": "011101", "OD": "04"},
    {"No": "50", "Gua": "䷱", "TC": "鼎", "Pinyin": "dǐng", "SC": "鼎", "English": "Establishing The New", "Binary": "101110", "OD": "03"},
    {"No": "51", "Gua": "䷲", "TC": "震", "Pinyin": "zhèn", "SC": "震", "English": "Taking Action", "Binary": "001001", "OD": "57"},
    {"No": "52", "Gua": "䷳", "TC": "艮", "Pinyin": "gèn", "SC": "艮", "English": "Keeping Still", "Binary": "100100", "OD": "58"},
    {"No": "53", "Gua": "䷴", "TC": "漸", "Pinyin": "jiàn", "SC": "渐", "English": "Developing Gradually", "Binary": "110100", "OD": "54"},
    {"No": "54", "Gua": "䷵", "TC": "歸妹", "Pinyin": "guī mèi", "SC": "归妹", "English": "Marrying Maiden", "Binary": "001011", "OD": "53"},
    {"No": "55", "Gua": "䷶", "TC": "豐", "Pinyin": "fēng", "SC": "丰", "English": "Abundance", "Binary": "001101", "OD": "59"},
    {"No": "56", "Gua": "䷷", "TC": "旅", "Pinyin": "lǚ", "SC": "旅", "English": "Travelling", "Binary": "101100", "OD": "60"},
    {"No": "57", "Gua": "䷸", "TC": "巽", "Pinyin": "xùn", "SC": "巽", "English": "Proceeding Humbly", "Binary": "110110", "OD": "51"},
    {"No": "58", "Gua": "䷹", "TC": "兌", "Pinyin": "duì", "SC": "兑", "English": "Joyful", "Binary": "011011", "OD": "52"},
    {"No": "59", "Gua": "䷺", "TC": "渙", "Pinyin": "huàn", "SC": "涣", "English": "Dispersing", "Binary": "110010", "OD": "55"},
    {"No": "60", "Gua": "䷻", "TC": "節", "Pinyin": "jié", "SC": "节", "English": "Restricting", "Binary": "010011", "OD": "56"},
    {"No": "61", "Gua": "䷼", "TC": "中孚", "Pinyin": "zhōng fú", "SC": "中孚", "English": "Innermost Sincerity", "Binary": "110011", "OD": "62"},
    {"No": "62", "Gua": "䷽", "TC": "小過", "Pinyin": "xiǎo guò", "SC": "小过", "English": "Little Exceeding", "Binary": "001100", "OD": "61"},
    {"No": "63", "Gua": "䷾", "TC": "既濟", "Pinyin": "jì jì", "SC": "既济", "English": "Already Fulfilled", "Binary": "010101", "OD": "64"},
    {"No": "64", "Gua": "䷿", "TC": "未濟", "Pinyin": "wèi jì", "SC": "未济", "English": "Not Yet Fulfilled", "Binary": "101010", "OD": "63"}
]

def generate_hexagram():
    """Pick a random I Ching hexagram from the list."""
    index = random.randrange(len(i_ching_names))  # Get a random index
    return i_ching_names[index], index  # Return both the hexagram and its index

def generate_frame_with_dynamic_height(response_text_lines, box_width_with_margins, hexagram, hex_number, bottom_details):
    """Generate ASCII frame with dynamic height based on the response text lines."""

    # ASCII art template's middle part
    middle_template = '''#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |          {}[░]  |  
'''
    middle_blank_template = '''#  |   |                               |                  |  
'''
    hex_solid_template = '''           ████████          
'''
    hex_broken_template = '''           ███  ███          
'''
    # Original lines that don't need to be altered
    original_lines = [
        "#                                              + ___ +     ",
        "#  + --[ETH-PHONE-NODE]------------------------| USB | -- +",
        "#  |    ⤥ WWW.CYPHERPUNKIRC.PATS.COOL          + --- +    |",
        "#  |----------------------------------------------------  |",
        "#  |               GND/RST2  [   ] [   ]                  |",
        "#  |              MOI2/SCK2  [   ] [   ]    P4T/PRINT[░]  |",
        "#  |               5V/MISO2  [   ] [   ]       A/SELF[░]  |",
        "#  |                                              REF[░]  |",
        "#  |   + ----------------------------- +          ARG[░]  |",
        "#  |   |     CYPHER PUNK IRC | GATE    |       BIP/39[░]  |",
        "#  |   |_______________________________|                  |",
        "#  |                                         ____________/",
        "#   \_______________________________________/            "
    ]
    # The specific layout structure for the puzzle lines
    puzzle_templates = [
        "#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |      LBTY/17[█]~~|",
        "#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |      EQTY/87[█]~~|",
        "#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |      FRAT/99[█]~~|",
        "#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |          GET[░]  |",
        "#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |            9[ ]  |",
        "#  |   | &&&&&&&&&&&&&&&&&&&&&&&&&&&&& |          WRD[░]  |",
    ]
    # Reverse the hexagram
    reversed_hexagram = hexagram[::-1]

    # Extract the generated hex for the puzzle lines and add corresponding lines
    puzzle_filled = []
    for line_template, hex_char in zip(puzzle_templates, hexagram):
        if hex_char == "0":
            line_filled = line_template.replace("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", "           " + hex_broken_template.strip() + "          ")
        else:
            line_filled = line_template.replace("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", "           " + hex_solid_template.strip() + "          ")
        puzzle_filled.append(line_filled)
    
    # Construct the middle part of the ASCII frame for the rest of the response text lines
    middle_parts = []
    for _ in range(len(response_text_lines) - 5):
        random_letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', k=3))
        formatted_string = middle_template.format(random_letters)
        middle_parts.append(formatted_string)

    # Create the bottom_details_filled section
    bottom_details_filled_parts = []
    for line in bottom_details:
        random_letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', k=3))
        formatted_string = middle_template.replace("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", line).format(random_letters)
        bottom_details_filled_parts.append(formatted_string)
    bottom_details_filled = ''.join(bottom_details_filled_parts) 

    # Replace the placeholders in the middle part with the response text lines
    middle_filled = ''.join(
        part.replace("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", line) 
        for part, line in zip(middle_parts, response_text_lines[0:])
    )

    # Display hex name here as a middle_filled line
    hex_info = next((item for item in i_ching_names if item["Binary"] == hexagram), None)
    hex_name = hex_info["English"] if hex_info else "Unknown Hexagram"
    hex_name_centered = hex_name.center(31)  # Centering based on the available space
    hex_name_line = middle_blank_template.replace("                               ", hex_name_centered)

    # Display hex number centered
    hex_number_display = "Hex #" + hex_number  # Use the actual hex number
    hex_number_centered = hex_number_display.center(31)
    hex_number_line = middle_blank_template.replace("                               ", hex_number_centered)   

    # Concatenate the parts to form the full ASCII frame
    full_ascii_frame = (
    "\n".join(original_lines[:11]) + "\n" +
    middle_blank_template +
    "\n".join(puzzle_filled) + "\n" +
    middle_blank_template + 
    hex_name_line +
    hex_number_line +
    middle_blank_template +
    middle_filled +
    middle_blank_template +
    bottom_details_filled +
    "\n".join(original_lines[10:])
)
    return full_ascii_frame

def divination():
    """Generate an ASCII frame with a futuristic interpretation of the I Ching."""
    hex_info, index = generate_hexagram()  # Call once to get both hexagram and index
    
    # Extract the specific details of the hexagram using the index
    hexagram = hex_info["Binary"]
    hex_number = hex_info["No"]

    # Getting OpenAI API key from .env file using config from decouple 
    openai.api_key = config("OPENAI_API_KEY")
    # Creating a prompt for the GPT-4 engine
    title = hex_info["English"]
    prompt = f"prompt for 0xSatoshi's response: Given I ching'{title}' #{hex_number}, a cypherpunk who helped our hero code this trading bot to aid him in his mission to take down the megacorps '0xSatoshi', our friend and ally '0xSatoshi' sits infront of a glowing blue terminal illuminated by the flashing lights, 0xSatoshi snaps out of trance, locks eyes with the hero, and and in his way recites the meaning of the given i ching hex saying: "
    # Query openai and store response
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256
    )
    response_text = response.choices[0].text.strip()  

    # Get the current real-time and format it as UTC for the year 2042
    current_time = datetime.utcnow().strftime('%H:%M:%S')
    timestamp = f"UTC:2042-05-12 {current_time}"

    # Formatting the response for ASCII art frame
    box_width_with_margins = 27
    response_text_lines = [
        ' ' + line.center(box_width_with_margins) + ' '
        for line in textwrap.wrap(response_text, width=box_width_with_margins)
    ]
    # Add the bottom details
    bottom_details = [
        "                             ",
        "      TRANSMISSIONS:LIVE     ",
        "       AUTHOR:0xSATOSHI      ",
        "       END:TRANSMISSION      ",
        "   " + timestamp + "   "
    ]
    # Extend the response_text_lines with the bottom details
    response_text_lines.extend(bottom_details)

    # Generate ASCII frame with dynamic height based on the response text lines
    ascii_frame = generate_frame_with_dynamic_height(response_text_lines, box_width_with_margins, hexagram, hex_number, bottom_details)
    return ascii_frame
result_dynamic = divination()
print(result_dynamic)
