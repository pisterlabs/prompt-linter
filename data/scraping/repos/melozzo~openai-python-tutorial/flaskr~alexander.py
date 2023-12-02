import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": ""
    },
    {
      "role": "assistant",
      "content": "what were Alexander the Great's most important battles\n"
    },
    {
      "role": "assistant",
      "content": "Alexander the Great fought several important battles during his conquests, but some of his most significant battles include:\n\n1. Battle of Granicus (334 BC): This was Alexander's first major battle against the Persian Empire. He led his army to victory and secured his control over Anatolia.\n\n2. Battle of Issus (333 BC): Alexander defeated the Persian King Darius III in this battle, which marked a turning point in his campaign against Persia. He captured Darius's family and gained control of Phoenicia and Egypt.\n\n3. Battle of Gaugamela (331 BC): This battle is considered one of Alexander's greatest victories. He defeated Darius III for the final time, secured his control over Persia, and expanded his empire to the east.\n\n4. Siege of Tyre (332 BC): Although not a battle in the strictest sense, this siege was a significant military operation. Alexander built a causeway to connect the island city of Tyre to the mainland and eventually conquered it after a seven-month-long siege.\n\n5. Battle of Hydaspes (326 BC): This battle took place in ancient India against King Porus. Despite facing a much larger opponent, Alexander emerged victorious and gained control over the Punjab region.\n\nThese battles played a"
    },
    {
      "role": "user",
      "content": "what are the geo coordinates of these battle locations\n"
    },
    {
      "role": "assistant",
      "content": "1. Battle of Granicus: The exact coordinates of the Battle of Granicus are not specified, but it took place in the vicinity of the Granicus River in modern-day Turkey. The approximate coordinates of the Granicus River are 39.448°N latitude, 27.081°E longitude.\n\n2. Battle of Issus: The Battle of Issus took place near the town of Issus in ancient Cilicia, which is now part of modern-day Turkey. The approximate coordinates of Issus are 36.707°N latitude, 35.926°E longitude.\n\n3. Battle of Gaugamela: The Battle of Gaugamela took place near the town of Gaugamela, which is now known as Tel Gomel in northern Iraq. The approximate coordinates of Tel Gomel are 36.774°N latitude, 43.084°E longitude.\n\n4. Siege of Tyre: Tyre is an island city located in modern-day Lebanon. The approximate coordinates of Tyre are 33.273°N latitude, 35.194°E longitude.\n\n5. Battle of Hydaspes: The Battle of Hydaspes took place near the Hydaspes River in ancient Punjab, which is now part of modern-day Pakistan. The approximate coordinates of the Hydaspes River are 33.363°N latitude, 75.082°E longitude.\n\nPlease note that these coordinates are approximate and may not be exact locations where the battles occurred."
    },
    {
      "role": "user",
      "content": "what modern city and country is nearest each of these battle sites\n"
    },
    {
      "role": "assistant",
      "content": "1. Battle of Granicus: The battle took place near the Granicus River in Turkey, and the modern city closest to the site is Çanakkale, Turkey.\n\n2. Battle of Issus: The battle occurred near the town of Issus, which is now known as İskenderun, in the Hatay province of Turkey.\n\n3. Battle of Gaugamela: The battle happened near the town of Gaugamela, which is now known as Arbil or Erbil in Iraq.\n\n4. Siege of Tyre: Tyre is located in modern-day Lebanon, and the nearest major city is the city of Tyre itself.\n\n5. Battle of Hydaspes: The battle took place near the Hydaspes River in ancient Punjab, which is now part of modern-day Pakistan. The nearest major city is Jhelum, Pakistan.\n\nPlease note that while these are the nearest modern cities to these battle sites, they may not be the exact cities closest to the specific locations where the battles occurred."
    }
  ],
  temperature=1,
  max_tokens=1472,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)