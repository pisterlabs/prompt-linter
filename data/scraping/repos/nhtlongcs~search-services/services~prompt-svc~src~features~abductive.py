import re 
from .base import InstructionBase 
from typing import List
import logging
import asyncio
import openai

class AbductiveChain(InstructionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chain = []
        

    def execute(self, query: str) -> List[str]:
        """
        Returns the top 5 most likely abductive explanations for the query
        """
        system_prompt = self.get_system_prompt()

        query = re.sub(r'[\n\t]', '', query)
        query = re.sub(r'\.$', '', query)
        query = query + '.'

        content = f"QUERY: {query}"
        result = super().execute(instruction=system_prompt, content=content, temperature=0.0) # deterministic
            

        result = re.sub(r'OUTPUT:|END.', '', result)
        result = result.strip()

        try:
            result = self._parse(result)
        except Exception as e:
            logging.warning(f"Failed to parse result: {e}, returning raw result.")
            result = [result]

        return result

    async def execute_wo_rate_limit(self, query: str) -> List[str]:
        try:
            return self.execute(query)
        except openai.error.RateLimitError as e:
            logging.info(f"Rate limit error: {e}, retrying in 1 second.")
            await asyncio.sleep(1)
            return self.execute_wo_rate_limit(query)
        
    def _parse(self, result):
        # match query from 1. to 5.
        result = re.findall(r'\d\.\s(.*)', result)
        result = [r.strip() for r in result]
        return result

    def get_system_prompt(self):
        
        system_prompt = self._get_instruction()
        system_prompt += "### These are good examples ###"
        system_prompt += '\n'.join(self._get_examples())
        system_prompt += "### Desired return format ###"
        system_prompt += self._get_format()
        
        return system_prompt
    
    def _get_instruction(self):

        return """
        You are a prompt suggester,
        provided sequence of clues, construct
        a semantic query that is embedded in latent space.
        Place the query in the most relevant and logical
        context based on your understanding, rephrase
        it into clear English. Avoid using abbreviations
        and ensure that it is suitable for searching in a
        vector database. Since all images in the database
        are captured from a first-person perspective, the
        query must be written in the passive voice.
        Combine the fragment hints, which are separated
        by commas, to form the most natural query possible.
        """

    def _get_examples(self):
        return [
            """
            QUERY: Preaching to a full room

            ABDUCTION:

            Preaching is typically associated with grand cathedrals, where passionate speakers deliver sermons to a captivated audience. 
            The church is often filled to capacity, with listeners fully focused on the speaker, who may be standing on the pulpit. 
            However, even outside of cathedral settings, powerful talks can be given in packed rooms, with speakers inspiring and 
            motivating their listeners. So these are my 5 modified versions 

            OUTPUT: 

            1. Delivering a passionate sermon to a packed congregation in a grand cathedral.

            2. Speaking to a captivated audience in a crowded auditorium, delivering a powerful message.

            3. Addressing a room filled with attentive listeners, and delivering an inspiring speech.

            4. Standing at the pulpit in front of a full house, delivering a moving sermon.

            5. Giving a rousing talk to a packed room, inspiring and motivating the audience.
            END.
            """,
            """

            QUERY: Preaching to a full room, hand appears in camera

            ABDUCTION:

            Preaching is often associated with grand cathedrals, where passionate speakers deliver sermons to a captivated audience. 
            The speaker may be standing on a pulpit, addressing a fully packed room filled with people who are focused on their 
            every word. However, preaching can also take place outside of the church, with speakers delivering rousing talks in 
            modern conference rooms equipped with the latest technology, such as cameras and microphones. The visible hand in the 
            camera may be reaching out toward the audience or holding a microphone, adding a modern touch to the presentation. 
            Teachers, professors, and presenters may also use projectors to point to slides while delivering lectures to their colleagues. 
            As a listener, it's important to keep track of the speech by taking notes or using a pen. So these are my 5 modified versions 

            OUTPUT: 

            1. A hand reaching out towards the camera, as if to emphasize a point during a speech to a full room.

            2. A hand holding a microphone, gesturing towards the audience in a crowded lecture hall.

            3. A hand waving to the crowd during a lively presentation to a packed conference room.

            4. A hand holding a pen, scribbling notes on a notepad while listening to a speaker in a full
            seminar.

            5. A hand pointing to a slide on a projector screen during a presentation to a full room of colleagues.
            END.
            """,
            """
            QUERY: Mother Mary, praying.

            ABDUCTION:

            Mother Mary is a central figure in the Christian faith, considered the mother of Jesus and revered as a symbol of maternal 
            love and devotion. Praying is a common practice among Christians, and Mother Mary is often depicted in art and iconography 
            with her hands clasped in prayer. This image evokes a sense of reverence and piety, as Mary is seen as a deeply spiritual 
            and holy figure. So these are my 5 modified versions

            OUTPUT:

            1. An image of Mother Mary with her hands clasped in prayer, evoking a sense of piety and devotion.

            2. A depiction of Mother Mary, the mother of Jesus, deep in prayer and contemplation.

            3. A statue of Mother Mary, with her eyes closed and her hands clasped in reverence.

            4. A painting of Mother Mary, kneeling in front of an altar and praying with humility and devotion.

            5. A symbol of the Christian faith, Mother Mary is often depicted in prayer, conveying a sense of spiritual devotion and love.
            
            END.
            """
        ]

    def _get_format(self):
        return """
        FORMAT: Return abduction and modified versions from the given query
        ABDUCTION: [ABDUCTION_HERE]
        OUTPUT: [THE_MODIFIED_QUERY]
        END.
        """