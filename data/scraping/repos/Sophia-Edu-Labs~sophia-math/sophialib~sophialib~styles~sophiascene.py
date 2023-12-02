from abc import ABC, abstractmethod
import os
from typing import List, Literal, Optional, Callable, Union as UnionType
from manim import *
from PIL import Image
from manim.mobject.mobject import Mobject
from manim.scene.scene import Scene
import numpy as np
import urllib.request
from pathlib import Path
import math
import re

from manim_voiceover import VoiceoverScene
from sophialib.constants.directories import SCENES_ASSETS_FOLDER
from sophialib.sounds.dynamic_sounds import StretchedAudioFileConfig, create_combined_temporary_stretched_audio_file_from_configs, create_temporary_stretched_audio_file, create_temporary_stretched_audio_file_from_config
from sophialib.styles.OpenAIVoiceoverService import OpenAIVoiceoverService
from sophialib.styles.styleconstants import *
from sophialib.styles.PollyVoiceoverService import AWSPollyService
from sophialib.styles.ElevenlabsVoiceoverService import ElevenlabsVoiceoverService
from sophialib.tasks.sophiataskdefinition import SophiaTaskDefinition
from sophialib.translation.currentlocale import CURRENT_LOCALE
from sophialib.translation.translate import get_translation
from sophialib.utils.xml import remove_xml_tags

# class EmojiSVGMobject(SVGMobject):
#     def __init__(self, emoji, **kwargs):
#         emoji_code = '-'.join(f'{ord(c):x}' for c in emoji)
#         url = f'https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/{emoji_code}.svg'
#         path_svg = Path.cwd() / f'{emoji_code}.svg'
#         urllib.request.urlretrieve(url, path_svg)
#         SVGMobject.__init__(self, str(path_svg), **kwargs)
#         path_svg.unlink()  # delete downloaded svg again locally

aspect_ratio = 9/16
sophia_radius_scale_factor_relative_to_height = 0.4*0.5
sophia_center_as_percentage_of_height = 1-sophia_radius_scale_factor_relative_to_height*0.8

# Define an "interface" that allows specific scenes to provide task defintion information
class SophiaQuestionInfo(ABC):
    @abstractmethod
    def task_definition(self) -> SophiaTaskDefinition:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if the subclass is also a subclass of SophiaScene
        if not issubclass(cls, SophiaScene):
            raise TypeError(f"{cls.__name__} must also inherit from SophiaScene!")


class SophiaScene(VoiceoverScene):
    should_add_dev_rects = True
    prevlocs = [RIGHT, RIGHT, RIGHT]
    cheight, cwidth = ValueTracker(0.2), ValueTracker(0.2)
    cx, cy = ValueTracker(0), ValueTracker(0)
    fill_opacity = 1
    blinking = 1

    def current_locale(self) -> str:
        return CURRENT_LOCALE

    def translate(self, key: str, fallback: Optional[str] = None) -> str:
        # use the translate helpers to search for the given key in the translation files
        translation = get_translation(key, self.current_locale())

        # if no translation was found, use the fallback if it was given, otherwise raise an error
        if translation is None:
            if fallback is not None:
                return fallback
            else:
                raise Exception(f"Could not find translation for key {key} in locale {self.current_locale()}")
            
        return translation


    def add_wrapped_subcaption(
        self,
        subcaption: str,
        duration: float,
        subcaption_buff: float = 0.1,
        max_subcaption_len: int = 70,
    ) -> None:
        subcaption_without_xml = remove_xml_tags(subcaption)
        super().add_wrapped_subcaption(subcaption_without_xml, duration, subcaption_buff, max_subcaption_len)

#Add Rectangles describing the Phone Screen
    def add_basic_rectangles(self):
        if not self.should_add_dev_rects:
            return
        main_rect = Rectangle(width=self.camera.frame_height*aspect_ratio, height=self.camera.frame_height, color=c_rects)
        self.add(main_rect)

#Add numbers for orientation in manim coordinates
    def add_orientation(self):
        
        for idx in range(-4,5,1):
            line = Line((-2.2,idx,-100),(2.2,idx,-100), color=c_or, stroke_width=1)
            t1 = Tex(f"{idx}", color=c_or).move_to((-3,idx,-100))
            t2 = Tex(f"{idx}", color=c_or).move_to((3,idx,-100))
            self.add(line, t1, t2)

        for idx in range(-2,3,1):
            line = Line((idx,-4,-100),(idx,4,-100), color=c_or, stroke_width=1)
            t1 = Tex(f"{idx}", color=c_or).move_to((idx, -3.5,-100))
            self.add(line, t1)

        tx = Text("X:", color=BLACK, font_size=fs3).move_to([-6,3,0])
        ty = Text("Y:", color=BLACK, font_size=fs3).move_to([-6,2,0])
        console = Text("Console", color=BLACK, font_size=fs3).move_to([-6,1,0])
        self.add(DecimalNumber(0, color=BLACK, font_size=fs3).add_updater(lambda m: m.set_value(self.cx.get_value())).next_to(tx, RIGHT, buff=0.2))
        self.add(DecimalNumber(1, color=BLACK, font_size=fs3).add_updater(lambda m: m.set_value(self.cy.get_value())).next_to(ty, RIGHT, buff=0.2))
        self.add(tx, ty, console)
        return console
            
#Add the mathgrid for notebook-look
    def add_mathgrid(self):
        # xrange = [2.2,6.6,0.2]
        xrange = [-2.2,11,0.2]
        yrange = [4,12,0.2]
        linestyledict = {"stroke_color": c_mathgrid, "stroke_width": 1}
        grid = NumberPlane(x_range=xrange, y_range=yrange, background_line_style=linestyledict, faded_line_style=linestyledict, x_axis_config=linestyledict, y_axis_config=linestyledict)
        self.add(grid)
        self.bring_to_back(grid)

    def add_divider(self):
        line = DashedLine([-2.2,-1,0], [2.2,-1,0], color=BLACK, stroke_width=1)
        self.add(line)

    def evaluate_string(self, s):
        # This regex pattern finds 'self.' followed by any word characters
        pattern_self = re.compile(r'self\.(\w+)')
        # Replace 'self.XXX' with 'XXX'
        s = pattern_self.sub(r'\1', s)
        
        # This regex pattern finds expressions enclosed in curly braces
        pattern_expr = re.compile(r'{(.*?)}')
        
        def replacer(match):
            # Evaluate the expression within the context of this class instance
            expression = match.group(1)
            return str(eval(expression, vars(self)))
        
        # Replace expressions in the string with their evaluated values
        evaluated_string = pattern_expr.sub(replacer, s)
        return evaluated_string

        

    def construct(self):
    
        # large debug print to debug which language this scene is rendered with
        print("##################################################################")
        print(f"\033[91mRendering scene {self.__class__.__name__} with locale \033[92m{self.current_locale()}\033[0m", flush=True)
        print("##################################################################", flush=True)

        self.add_basic_rectangles()
        self.camera.background_color = c_bg
        
        
        #if the environmentvariable SPEECH_SERVICE is set, always use this service, otherwise the defeault specified below
        if os.environ.get('SPEECH_SERVICE') == 'aws':
            speechs = AWSPollyService()
        elif os.environ.get('SPEECH_SERVICE') == 'openai':
            speechs = OpenAIVoiceoverService()
        elif os.environ.get('SPEECH_SERVICE') == 'elevenlabs':
            speechs = ElevenlabsVoiceoverService(should_use_history=True)
        else:
            ##################################################################
            ##################################################################
            ##################################################################
            ################# Manually modify? BELOW!!!! #####################
            ##################################################################
            ##################################################################
            ##################################################################
            speechs = ElevenlabsVoiceoverService(should_use_history=True)
            # optionally use this. 
            # speechs = AWSPollyService()
        
        # set speech service
        self.set_speech_service(speechs)

    def voiceover(self, text: str = None, ssml: str = None, **kwargs):
        if (len(text) > 2145 and not "subcaption" in kwargs): # noone knows from where this number comes from, but it seems to be the (a?) limit
            raise Exception("Try to call voiceover with subcaption='NOT AVAILABLE DUE TO BUG' (no empty value!) again, because the text is too long which raises some exceptions in manim-voiceover, which seems to be a bug with manim-voiceover…")
        return super().voiceover(text, ssml, **kwargs)
    
    #Functions used in Scenes:
    #Movement-Related

    #Smoothly move an object to a new position
    #TO BE REMOVED
    def move_smooth(self, obj, end_point, run_time=1.2):
        start_point=obj.get_center()
        start_point[2] = end_point[2]
        #curve = CurvedArrow(start_point=start_point, end_point=end_point)
        curve = Line(start_point, end_point, color=c1)
        self.play(MoveAlongPath(obj, curve), rate_func=smooth, run_time=run_time)


    #TO BE REMOVED
    def draw_point(self, loc, object, offset, color=c1, rate_func=smooth):
        loc, offset = np.array(loc), np.array(offset)
        self.move_smooth(object, loc+offset, run_time=0.5)
        point  = Dot(color=color).move_to(loc)
        object.add_updater(lambda d: d.move_to(point.get_center()+offset+fg))
        self.play(Write(point))
        object.clear_updaters()
        return point
    
    #TO BE REMOVED
    def fly_out(self, object, run_time=1):
        self.move_smooth(object, outloc, run_time=run_time)


    def pulse(self, object, run_time=1, factor=2):
        self.play(ScaleInPlace(object, factor), run_time=run_time/2)
        self.play(ScaleInPlace(object, 1/factor), run_time=run_time/2)

    #TO BE REMOVED
    def move_curved(self, t1, t2, dest, depth=0.25, run_time=0.4):
                o1, o2 = t1.get_value(), t2.get_value()
                dummy = Circle(stroke_width=0)
                self.add(dummy) 
                def updater(x):
                    eps = 0.0001
                    t2.set_value(o2 -depth*(1-(2*(t1.get_value() - (eps+dest+o1)/2)/(eps+dest-o1))**2))
                dummy.add_updater(updater)
                self.play(t1.animate(run_time=run_time).set_value(dest))
                self.remove(dummy)

    #TO BE REMOVED
    def circle(self, radius, iterations=3, run_time=1.5):
        x,y = self.cx.get_value()-radius, self.cy.get_value()
        angle = ValueTracker(0)
        self.add(angle)
        circ = Circle(stroke_width=0).add_updater(lambda m: self.cx.set_value(x+radius*np.cos(angle.get_value()))).add_updater(lambda m: self.cy.set_value(y+radius*np.sin(angle.get_value())))
        self.add(circ)
        for idx in range(iterations):
            self.play(angle.animate.set_value(2*(idx+1)*PI), run_time=run_time, rate_func=linear)
        self.remove(circ, angle)




        
    #TO BE REMOVED
    #Creating Things
    def draw_line(self, a, b, object, offset, color=GREY, run_time=2, rate_func=smooth, linetype="Line"):
        a, b, offset = np.array(a), np.array(b), np.array(offset)
        self.move_smooth(object, a+offset, run_time=0.5)
        path = Line(a+offset+fg, b+offset+fg)
        line = VMobject().move_to(a)
        self.add(line)
        if linetype=="Line":
            line.add_updater(lambda x: x.become(Line(a, object.get_center()-offset-fg, color=color)))
        elif linetype=="DashedLine":
            line.add_updater(lambda x: x.become(DashedLine(a, object.get_center()-offset-fg, dash_length=0.05, dashed_ratio=0.5, color=color)))
        self.play(MoveAlongPath(object, path), rate_func=rate_func, run_time=run_time)
        line.clear_updaters()
        if linetype=="Line":
            line2 = Line(path.get_start()-fg, path.get_end()-fg, color=color).move_to(line)
        elif linetype=="DashedLine":
            line2 = DashedLine(path.get_start()-fg, path.get_end()-fg, dash_length=0.05, dashed_ratio=0.5, color=color).move_to(line)
        self.add(line2)
        self.remove(line)
        return line2
    
    #TO BE REMOVED
    def draw_func(self, cords, func, object, offset, color=c1, run_time=2, stroke_width=3, rate_func=linear):
        object.add_updater(lambda m: m.move_to(func.get_end()+offset))
        func = cords[0].plot(func, color=color, stroke_width=stroke_width)
        self.play(Write(func), run_time=run_time, rate_func=rate_func)
        object.clear_updaters()
        return func
        



    def add_title(self, title="Title of a SophiaScene", tex=True):
        font_size = fs1 if len(title) < 12 else (fs2 if len(title) < 20 else fs3)
        title = Text(title, font_size=font_size, color=c1, font=font_0).to_edge(UP) if not tex else Tex(title, font_size=font_size, color=c1).to_edge(UP)
        title = title.scale(min(3.2/title.get_width(), 0.4/title.get_height()))

        remainder_title = title.get_bottom()[1] % 0.2
        if remainder_title != 0:
            if remainder_title < 0.1:
                title.shift(DOWN * remainder_title)
            else:
                title.shift(UP * (0.2 - remainder_title))

        #self.add(title)
        return title
        
    def add_cords(self, x_range = [-1,1,1], y_range = [-1,1,1], x_ticks=[-1,0,1], y_ticks=[-1,0,1], x_labels = None, y_labels = None, width=2.4, height=2.4, axisLabelX='x', axisLabelY='y', tip_x=StealthTip, tip_y=StealthTip, include_tip = True):

        if x_labels is None:
            x_labels = x_ticks
        if y_labels is None:
            y_labels = y_ticks
            

        cords = VGroup()
        num_x_ticks = len(x_ticks) - 1
        num_y_ticks = len(y_ticks) - 1
        max_ticks = max(num_x_ticks, num_y_ticks)
        unit_size_x = width / num_x_ticks
        unit_size_y = height / num_y_ticks
        x_length = num_x_ticks * unit_size_x
        y_length = num_y_ticks * unit_size_y

        # Adjust y_length to ensure y-axis has length 2.4
        if max_ticks == num_y_ticks:
            y_length = height

        plane = NumberPlane(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, background_line_style={"stroke_opacity": 0}, axis_config={"include_tip": include_tip, 'tip_width': 0.15, 'tip_height': 0.1}).move_to(np.array([0,1.4,0]))
        plane.color=BLACK
        plane.x_axis.add_ticks()
        plane.y_axis.add_ticks()
        #plane.unit_size = unit_size
        cords.add(plane)
        if x_range[0] <= 0:
            cords.add(Tex('0', color=c1, font_size=fs3).move_to(plane.x_axis.n2p(0)).shift(0.2*DOWN+0.2*RIGHT))

        for idx, tick in enumerate(x_ticks):
            if tick != 0:
                cords.add(MathTex(x_labels[idx], color=c1, font_size=fs3).move_to(plane.x_axis.n2p(tick)).shift(0.2*DOWN))
        for idx, tick in enumerate(y_ticks):
            if tick != 0:
                cords.add(MathTex(y_labels[idx], color=c1, font_size=fs3).move_to(plane.y_axis.n2p(tick)).shift(0.2*LEFT))

        cords.add(plane.get_x_axis_label(Tex(axisLabelX, color=c1, font_size=fs3))) if axisLabelX is not None else None
        cords.add(plane.get_y_axis_label(Tex(axisLabelY, color=c1, font_size=fs3))) if axisLabelY is not None else None
        # if axisLabelX is not None: 
        #     xlabel = Tex(axisLabelX, color=c1t, font_size=fs3).next_to(plane.x_axis, DOWN, buff=0.2)
        #     xlabel = xlabel.scale(min(1,0.6*plane.x_axis.get_width()/xlabel.get_width()))
        #     cords.add(xlabel)
        # if axisLabelY is not None:
        #     ylabel = Tex(axisLabelY, color=c1t, font_size=fs3).next_to(plane.y_axis, UP, buff=0.2)
        #     ylabel = ylabel.scale(min(1,0.6*plane.y_axis.get_height()/ylabel.get_height()))
        #     cords.add(ylabel)

            # Adjust x_axis position
        remainder_x = plane.x_axis.get_y() % 0.2
        if remainder_x != 0:
            if remainder_x < 0.1:
                cords.shift(DOWN * remainder_x)
            else:
                cords.shift(UP * (0.2 - remainder_x))

        # Adjust y_axis position
        remainder_y = plane.y_axis.get_x() % 0.2
        if remainder_y != 0:
            if remainder_y < 0.1:
                cords.shift(LEFT * remainder_y)
            else:
                cords.shift(RIGHT * (0.2 - remainder_y))

        return cords


    #TO BE REMOVED
    def equation_parts(self, parts, colors=c1, font_size=fs2, **kwargs):
        if type(colors)==str or type(colors)==type(rgb_to_color([0,0,0])):
            colors = [colors]*len(parts)
        elements = [MathTex(parts[0], color=colors[0], font_size=font_size, **kwargs) if type(parts[0])==str else parts[0]]
        for idx, part in enumerate(parts[1:]):
            if type(part)==str:
                elements.append(MathTex(part, color=colors[1+idx], font_size=font_size, **kwargs).next_to(elements[-1], 0.35*RIGHT))
            elif type(part)==Tex or type(part)==DecimalNumber:
                elements.append(part.next_to(elements[-1], 0.35*RIGHT))
            else:
                raise Exception("Type of part not supported")
        return VGroup(*elements)#, elements
    

    def transition_to_question(self, question, choices, idx_logged=0, idx_correct=0, fontsizeQ=fs2, fontsizeChoices=fs2):
        all_elements = VGroup(*[x for x in self.mobjects if isinstance(x, VMobject)][2:])
        for element in all_elements:
            element.clear_updaters()

        rectangles=VGroup(*[RoundedRectangle(height=0.8, width=3.5, corner_radius=0.2, color=c1, fill_opacity=0.2).move_to([0, -10-choices.index(choice), 0]) for choice in choices])
        rectangles.add(*[MathTex(choices[idx], color=c1, font_size=fontsizeChoices).move_to(rectangles[idx]) for idx in range(len(choices))])
        self.add(rectangles)

        q = Tex(question, color=c1, font_size=fontsizeQ).move_to([-10,3.2,0])

        self.play(rectangles.animate.shift(UP*10), all_elements.animate.shift(UP*10), q.animate.shift(RIGHT*10), run_time=1.5)
        self.wait(5)
        self.play(rectangles[idx_logged].animate.set_color(YELLOW), run_time=0.5)
        self.wait(2)
        if idx_logged != idx_correct:
            self.play(rectangles[idx_logged].animate.set_color(RED), rectangles[idx_correct].animate.set_color(GREEN), run_time=1)
        else:
            self.play(rectangles[idx_logged].animate.set_color(GREEN), run_time=1)

            
    # The desired final dimensions of the square
    def redraw_square(self):
        newloc = RIGHT*(self.cx.get_value()+self.cwidth.get_value()/2)+UP*(self.cy.get_value()+self.cheight.get_value()/2)
        self.prevlocs = [newloc, self.prevlocs[0], self.prevlocs[1]]
        rect = RoundedRectangle(color=RED, height=self.cheight.get_value(), width=self.cwidth.get_value(), corner_radius=0.1,
                            fill_opacity = self.fill_opacity, stroke_width=0).shift(newloc)
        return rect

    #TO BE REMOVED
    def add_cursor(self, color=RED, bg_opacity=0.4):
            self.cheight, self.cwidth = ValueTracker(0.2), ValueTracker(0.2)
            self.cx, self.cy = ValueTracker(0), ValueTracker(0)
            self.prevlocs = [RIGHT, RIGHT, RIGHT]


            def updater(mob, dt):
                if max(mob.height, mob.width)>=0.201:
                    self.fill_opacity = max(bg_opacity, math.exp(-2.5 * (max(mob.height, mob.width) - bg_opacity)))
                    self.bring_to_back(mob)
                    return
                if self.blinking==0 or not np.array_equal(self.prevlocs[0],self.prevlocs[1]) or not np.array_equal(self.prevlocs[1],self.prevlocs[2]):
                    self.fill_opacity = 1
                    mob.time_passed = 0
                    self.bring_to_front(mob)
                    return
                mob.time_passed += dt
                if mob.time_passed >= 0.6:
                    mob.time_passed -= 0.6
                    self.bring_to_front(mob)
                    if self.blinking==2:
                        self.fill_opacity = 1 if self.fill_opacity==0 else 0
                    else:
                        1 if self.fill_opacity ==0 else 0


            updated_square = Square()
            updated_square = always_redraw(self.redraw_square)
            updated_square.time_passed = 0
            updated_square.add_updater(updater)

            return updated_square
    
    ###################
    ####Sounds#########
    ###################
    def _get_sound_file(self, sound_file_name: str) -> str:
        return str(SCENES_ASSETS_FOLDER / "sounds" / f"{sound_file_name}.wav")
    
    def add_stretched_sound(self, sound_name: str, target_duration: float, target_volume: float = 1.0, reverse: bool = False, rate_func: Optional[Callable[[float],float]] = None, repeat_count: Optional[int] = 0):
        print(f"Adding stretched sound {sound_name} with target duration {target_duration}", flush=True)
        # create the temporary stretched version of the sound file
        temp_audio_file = create_temporary_stretched_audio_file(self._get_sound_file(sound_name), target_duration, target_volume, reverse, rate_func, repeat_count)

        # add the stretched sound to the scene
        self.add_sound(temp_audio_file)

    def add_stretched_sounds(self, sound_configs: List[StretchedAudioFileConfig]): 
        # if a single sound config input file is not a path, then it is a sound name so we need to convert it to a path using the _get_sound_file method
        for sound_config in sound_configs:
            if not isinstance(sound_config.input_file, Path):
                sound_config.input_file = self._get_sound_file(sound_config.input_file)
        temp_audio_file = create_combined_temporary_stretched_audio_file_from_configs(sound_configs)
        self.add_sound(temp_audio_file)


    def add_scale_sound(self, direction: UnionType[Literal["up"], Literal["down"]], duration=0.5, rate_func=smooth):
        self.add_stretched_sound("swipe-zoom-short", duration, target_volume=0.06, reverse=(direction=="down"), rate_func=rate_func)
    
    def add_move_sound(self, direction: UnionType[Literal["forward"], Literal["backward"]], duration=0.5, rate_func=smooth, target_volume=0.3):
        self.add_stretched_sound("jump", duration, target_volume=target_volume, reverse=(direction=="backward"), rate_func=rate_func)

    def add_reflect_sound(self, duration=0.5, rate_func=smooth):
        self.add_stretched_sound("huuup", duration, target_volume=0.1, rate_func=rate_func)

    def add_shift_sound(self, duration=0.5, rate_func=smooth, volume=1):
        self.add_stretched_sound("shift", duration, target_volume=0.8*volume, rate_func=rate_func )

    def add_cursor_sound(self, duration=0.3, rate_func=smooth):
        self.add_stretched_sound("cursor", duration, target_volume=0.2, rate_func=rate_func )

    def add_pencil_sound(self, duration=0.5, rate_func=smooth, volume=1):
        self.add_stretched_sound("pencil", duration*0.6, target_volume=volume*0.4, rate_func=rate_func )

    def add_bubble_sound(self, duration=0.5, rate_func=smooth):
        self.add_stretched_sound("bubble", duration, target_volume=0.3, rate_func=rate_func )

    ###################
    ####Drawings#######
    ###################
    def draw_smile(self, cursor, loc, happy=True, run_time=1.5):

        sq = lambda : Square(side_length=0.2, color=RED, fill_color=RED, fill_opacity=1).shift([-0.1,-0.1,0])
        c = lambda : Circle(radius=0.1, fill_color=RED, fill_opacity=1, color=RED).shift([-0.1,-0.3,0])

        eL = c().shift(LEFT*0.7)
        eR = c().shift(RIGHT*0.9)
        mouth = ArcBetweenPoints(LEFT+DOWN, RIGHT+DOWN, stroke_color=RED, stroke_width=12, angle=(2*(happy)-1)*2/3*PI).shift(DOWN*0.6*(1-happy))
        smile = VGroup(eL, eR, mouth).move_to(loc)
        circ = c().move_to(cursor.get_center())
        self.add(circ)
        cursorCopy = cursor.copy()
        self.remove(cursor)
        self.play(circ.animate.move_to(eL), run_time=0.2*run_time)
        self.add(eL)
        self.play(circ.animate.move_to(eR), run_time=0.2*run_time)
        self.add(eR)
        self.play(circ.animate.move_to(mouth.get_start()), run_time=0.2*run_time)
        circ.add_updater(lambda x: x.move_to(mouth.get_end()))
        self.play(Create(mouth), run_time=0.2*run_time)
        self.add(cursorCopy.move_to(circ))
        self.remove(circ)
        self.add(cursorCopy)
        return cursorCopy, smile
    
    def draw_qmark(self, cursor, loc=ORIGIN, run_time=1.5, apply_waves=2, wave_time=1):

        sq = lambda : Square(side_length=0.2, color=RED, fill_color=RED, fill_opacity=1).shift([-0.1,-0.1,0])
        c = lambda : Circle(radius=0.1, fill_color=RED, fill_opacity=1, color=RED).shift([-0.1,-0.3,0])
        Qmark = VGroup(sq().shift(UP*0.8+LEFT*0.2), sq().shift(UP), sq().shift(UP*0.8+RIGHT*0.2), sq().shift(UP*0.6+RIGHT*0.2), sq().shift(UP*0.4), sq().shift(UP*0.3), c().shift([0,-0.1,0])).move_to(loc)
        
        self.play(cursor.animate.move_to(Qmark[0]), run_time=0.2*run_time)
        circ = c().move_to(cursor)
        self.add(circ)

        for idx in range(6):
            self.play(ReplacementTransform(circ, Qmark[idx]), run_time=0.1*run_time)
            circ = c().move_to(Qmark[idx])
        
        self.play(circ.animate.move_to(Qmark[-1]), run_time=0.2*run_time)

        for _ in range(apply_waves):
            self.play(ApplyWave(circ), run_time=wave_time)

        
        cursor.move_to(Qmark[-1])
        self.remove(circ)

        return Qmark
    
    def draw_decorative_line(self, cursor, downshift = 0, run_time = 2):

        sq = lambda : Square(side_length=0.2, color=RED, fill_color=RED, fill_opacity=1, stroke_width=0)
        c = lambda : Circle(radius=0.1, fill_color=RED, fill_opacity=1, color=RED, stroke_width=0).shift([-0.1,-0.3,0])

        decoration1 = VGroup(*[sq().shift([-2.1+0.4*idx,0.1,0]) for idx in range(12)]).shift(DOWN*downshift)
        decoration2 = VGroup(*[sq().shift([-1.9+0.4*idx,-0.1,0]) for idx in range(12)]).shift(DOWN*downshift)
        decoration = VGroup(*decoration1, *decoration2)

        cursor.blinking=False
        self.play(cursor.animate.move_to(decoration1[0]), run_time=0.2*run_time)

        circ = c().move_to(decoration1[0])
        self.add(circ)
        cursor.move_to(decoration2[-1])
        for idx in range(12):
            self.play(ReplacementTransform(circ, decoration1[idx]), run_time=0.1*run_time)
            self.play(ReplacementTransform(c().move_to(decoration1[idx]), decoration2[idx]), run_time=0.1*run_time)
            circ = c().move_to(decoration2[idx])

        cursor.move_to(decoration1[-1])

        return decoration
        

    

# class Cursor(Mobject):
#     def __init__(self, scene, color=RED, bg_opacity=0.4, blinking=True, stroke_width=0, fill=True, **kwargs):
#         super().__init__(**kwargs)
#         self.scene = scene
#         self.cheight = ValueTracker(0.2)
#         self.cwidth = ValueTracker(0.2)
#         self.cx = ValueTracker(0)
#         self.cy = ValueTracker(0)
#         self.fill_opacity = 1
#         self.blinking = blinking
#         self.color = color
#         self.bg_opacity = bg_opacity
#         self.stroke_width = ValueTracker(stroke_width)
#         self.fill = fill
#         self.override_fill = ValueTracker(-1)
#         self.force_foreground=False

#         #This function updates the values from which the rectangle is built
#         def updater(mob, dt):
#             if self.force_foreground:
#                 scene.bring_to_front(mob)
#             #Deal with Larger Rectangles: Opacity depends on size
#             if self.override_fill.get_value() != -1:
#                 self.fill_opacity = self.override_fill.get_value()
#                 return
#             if max(mob.height, mob.width) >= 0.201:
#                 self.fill_opacity = max(self.bg_opacity, math.exp(-2.5 * (max(mob.height, mob.width) - self.bg_opacity)))
#                 if not self.force_foreground:
#                     scene.bring_to_back(mob)
#                 return
#             #If it doesn't blink, it's filled
#             if not self.blinking:
#                 self.fill_opacity = 1
#                 mob.time_passed = 0
#                 scene.bring_to_front(mob)
#                 return
            
#             else:
#                 mob.time_passed += dt
#                 if mob.time_passed >= 0.6:
#                     mob.time_passed -= 0.6
#                     scene.bring_to_front(mob)
#                     self.fill_opacity = 1 if self.fill_opacity == 0 else 0

#         self.square = Square()
#         self.square = always_redraw(self.redraw_square)
#         self.square.time_passed = 0
#         self.square.add_updater(updater)

#         self.add(self.square)

#     def redraw_square(self):
#         rect = RoundedRectangle(
#             color=self.color,
#             height=self.cheight.get_value(),
#             width=self.cwidth.get_value(),
#             corner_radius = min(0.1, min(self.cheight.get_value(), self.cwidth.get_value())/2),
#             fill_opacity=self.fill_opacity*self.fill,
#             stroke_width=self.stroke_width.get_value()
#         ).move_to([self.cx.get_value(), self.cy.get_value(), 0])
#         return rect
    
    #DEPRECATED, DO NOT USE
    # def move_curved(self, t1, t2, dest, depth=0.25, run_time=0.4):
    #     o1, o2 = t1.get_value(), t2.get_value()
    #     dummy = Circle(stroke_width=0)
    #     self.add(dummy) 
    #     def updater(x):
    #         eps = 0.0001
    #         t2.set_value(o2 -depth*(1-(2*(t1.get_value() - (eps+dest+o1)/2)/(eps+dest-o1))**2))
    #     dummy.add_updater(updater)
    #     self.scene.play(t1.animate(run_time=run_time).set_value(dest))
    #     self.remove(dummy)

    # def curved_updater(self, dest, t1=None, t2=None, depth=0.25):
    #     if t1==None:
    #         t1 = self.cx
    #     if t2==None:
    #         t2 = self.cy
    #     o1, o2 = t1.get_value(), t2.get_value()
    #     dummy = Circle(stroke_width=0)
    #     self.scene.add(dummy) 
    #     def updater(x):
    #         eps = 0.0001
    #         t2.set_value(o2 -depth*(1-(2*(t1.get_value() - (eps+dest+o1)/2)/(eps+dest-o1))**2))
    #     dummy.add_updater(updater)
    #     return dummy


    # def circle_updater(self, radius=0.2):
    #     ## USAGE EXAMPLE: ##
    #     # circ, angle = cursor.circle_updater(0.35)
    #     # for idx in range(3):
    #     #     self.play(angle.animate.set_value(2*(idx+1)*PI), run_time=0.5, rate_func=linear)
    #     # self.remove(circ, angle)
    #     ##  END OF EXAMPLE ##
    #     x,y = self.cx.get_value()-radius, self.cy.get_value()
    #     angle = ValueTracker(0)
    #     self.scene.add(angle)
    #     circ = Circle(stroke_width=0).add_updater(lambda m: self.cx.set_value(x+radius*np.cos(angle.get_value()))).add_updater(lambda m: self.cy.set_value(y+radius*np.sin(angle.get_value())))
    #     self.scene.add(circ)
    #     return circ, angle
    
class AltCursor(RoundedRectangle):
    DEFAULT_SIZE = 0.2
    DEFAULT_CURSOR_OPACITY = 1.0
    DEFAULT_CURSOR_OPACITY_IN_BACKGROUND = 0.4
    def __init__(self, 
                 width=DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 corner_radius=DEFAULT_SIZE/2.0,
                 x=0,
                 y=0,
                 color=RED, 
                 fill_opacity=DEFAULT_CURSOR_OPACITY, 
                 blinking=False,
                 stroke_width=0,
                 idle = False,
                 autoFadeBackground = True,
                 **kwargs):
        
        super().__init__(corner_radius=corner_radius, 
                         height=height, 
                         width=width, 
                         color=color, 
                         fill_opacity=fill_opacity,  
                         stroke_width=stroke_width,
                         **kwargs)
        self.move_to([x,y,0])
        self.blinking = blinking
        self.initial_fill_opcacity = fill_opacity
        # self._setup_blinking()
        self._setup_idle()
        self.autoFadeBackground = autoFadeBackground
        self.needSound = False
        self.idle = idle
    
    def animationTo(self, dest, run_time=1, rate_func=linear):
        return ApplyMethod(self.move_to, dest, run_time=run_time, rate_func=rate_func)

    def changeForm(self, 
                width = None, 
                height = None, 
                corner_radius = None,
                x = None,
                y = None,
                fill_opacity = None, 
                color = None, 
                stroke_width = None,
                idle = None,
                **kwargs
                ):
        if corner_radius is None:
            corner_radius = self.corner_radius
        if x is None:
            x = self.get_center()[0]
        if y is None:
            y = self.get_center()[1]
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if fill_opacity is None:
            fill_opacity = self.fill_opacity
        if self.autoFadeBackground and fill_opacity!=0 and not self.idle:
            fill_opacity = max(self.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND, math.exp(-2.5 * (max(height, width) - self.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND)))
        if color is None:
            color = self.color
        if stroke_width is None:
            stroke_width = self.stroke_width
        if idle is None:
            idle = self.idle


        newCursor = AltCursor(x=x, 
                              y=y, 
                              width=width,
                              height=height,
                              corner_radius=corner_radius,
                              fill_opacity=fill_opacity, 
                              color=color,
                              blinking=self.blinking,
                              stroke_width=stroke_width,
                              idle = self.idle,
                              **kwargs,
                              )
        newCursor.initial_fill_opcacity = self.initial_fill_opcacity
        self.become(newCursor)
        return newCursor

        

    def animationToForm(self, 
                        width,
                        height, 
                        corner_radius,
                        x = None,
                        y = None,
                        fill_opacity = None, 
                        color = None, 
                        ):
        return ApplyMethod(self.changeForm, width, height, corner_radius, x, y, fill_opacity, color)
    
    def setBlinking(self, blinking):
        self.blinking = blinking

    def _setup_blinking(self):
        self.time_passed = 0
        self.isHidden = False

        def _blinking_updater(mob: AltCursor, dt: float):

            if mob.blinking:
                mob.time_passed += dt
                if mob.time_passed >= 0.6:
                    mob.time_passed -= 0.6
                    mob.isHidden = not mob.isHidden
                    if not mob.isHidden:
                        self.needSound = True
                    mob.changeForm(fill_opacity=0 if mob.isHidden else mob.initial_fill_opcacity)

            elif mob.isHidden: # reset not required if not blinking and already visible
                mob.changeForm(fill_opacity=mob.initial_fill_opcacity)
                mob.time_passed = 0
                mob.isHidden = False
        self.add_updater(_blinking_updater)


    def set_idle(self, idle):
        self.idle = idle
    
    def _setup_idle(self):
        self.time_passed = 0

        last_idle_state = [False]  # Using a list to store the state

        def _idle_updater(mob: AltCursor, dt: float):
            idle_interval = 2  # Adjust as per your requirement

            if mob.idle:
                if not last_idle_state[0]:  # i.e., if mob.idle has been changed from False to True
                    mob.time_passed = 0  # Resetting time_passed as mob.idle is True now
                mob.time_passed += dt  # Incrementing time_passed
                # Calculate the new fill_opacity
                new_opacity = max(c_bg_opacity, c_fg_opacity - (c_fg_opacity - c_bg_opacity) * (mob.time_passed / idle_interval))
                mob.changeForm(fill_opacity=new_opacity)
            else:
                mob.changeForm(fill_opacity=c_fg_opacity)

            # Update the last_idle_state list with the current idle state of the mob.
            last_idle_state[0] = mob.idle



        self.add_updater(_idle_updater)





    def _start_fading(self, time):

        def _fade_updater(mob, dt):
            mob.fade_passed += dt
            mob.set_opacity(max(0,1-mob.fade_passed/mob.fade_time))

        self.fade_time = time
        self.fade_passed = 0
        self.blinking=False
        self.add_updater(_fade_updater)

        return self


class CursorPositionTracker:
    def __init__(self, cursor: AltCursor, pos_func: Callable[[], List[float]]):
        self.cursor = cursor
        self.pos_func = pos_func

    def __enter__(self):
        def update_cursor_position(mob, dt):
            pos = self.pos_func()
            mob.changeForm(x=pos[0], y=pos[1])

        self.updater = update_cursor_position
        self.cursor.add_updater(self.updater)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.remove_updater(self.updater)
    
class CursorMoveTo(ApplyMethod):
    def __init__(self, cursor, x, y, run_time=1, rate_func=smooth, **kwargs):
        if x is None:
            x = cursor.get_center()[0]
        if y is None:
            y = cursor.get_center()[1]
        
        super().__init__(cursor.move_to, [x,y,cursor.get_center()[2]], run_time=run_time, rate_func=rate_func, **kwargs)
        self.cursor = cursor
        self.x = x
        self.y = y

class CursorMoveToCurved(CursorMoveTo):
    def __init__(self, cursor, x, y, run_time=1, path_arc=PI/2, rate_func=smooth,  **kwargs):

        if cursor.get_x() > x:
            path_arc *= -1
        super().__init__(cursor, x, y, run_time=run_time, rate_func=rate_func, path_arc=path_arc, **kwargs)
        self.cursor = cursor
        self.x = x
        self.y = y

class CursorMarkAxis(ApplyMethod):
    def __init__(self, cursor: AltCursor, plane, axis='x', autoFadeBackground=False, moveToBackground=False, **kwargs):

        if axis == 'x':
            x,y,_ = plane.x_axis.get_center()
            width = plane.x_axis.get_width()+0.2
            height = 0.3
        elif axis == 'y':
            x, y, _ = plane.y_axis.get_center()
            width = 0.3
            height = plane.y_axis.get_height()+0.2
        else:
            raise ValueError("axis must be 'x' or 'y'")
        self.cursor = cursor
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.autoFadeBackground = autoFadeBackground
        self.moveToBackground = moveToBackground

        self.fill_opacity = cursor.DEFAULT_CURSOR_OPACITY
        if self.autoFadeBackground and max(self.height, self.width) >= 0.201:
            self.fill_opacity = max(cursor.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND, math.exp(-2.5 * (max(self.height, self.width) - cursor.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND)))
        
        super().__init__(cursor.changeForm, width, height, None, x, y, self.fill_opacity, None, None, **kwargs)

class CursorUnderline(ApplyMethod):
    def __init__(self, cursor: AltCursor, target:Mobject, autoFadeBackground=False, moveToBackground=False, **kwargs):

        x,y,_ = target.get_bottom()
        y -= 0.1
        width = target.get_width()+0.2
        height = 0.1

        self.cursor = cursor
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.autoFadeBackground = autoFadeBackground
        self.moveToBackground = moveToBackground

        self.fill_opacity = cursor.DEFAULT_CURSOR_OPACITY
        if self.autoFadeBackground and max(self.height, self.width) >= 0.201:
            self.fill_opacity = max(cursor.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND, math.exp(-2.5 * (max(self.height, self.width) - cursor.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND)))
        
        super().__init__(cursor.changeForm, width, height, None, x, y, self.fill_opacity, None, None, **kwargs)

class CursorMoveResize(ApplyMethod):
    def __init__(self, cursor: AltCursor, x,y, width=0.2, height=0.2, autoFadeBackground=False, moveToBackground=False, **kwargs):
        if x is None:
            x = cursor.get_center()[0]
        if y is None:
            y = cursor.get_center()[1]
        self.cursor = cursor
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.autoFadeBackground = autoFadeBackground
        self.moveToBackground = moveToBackground

        self.fill_opacity = cursor.DEFAULT_CURSOR_OPACITY
        if self.autoFadeBackground and max(self.height, self.width) >= 0.201:
            self.fill_opacity = max(cursor.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND, math.exp(-2.5 * (max(self.height, self.width) - cursor.DEFAULT_CURSOR_OPACITY_IN_BACKGROUND)))
        
        super().__init__(cursor.changeForm, width, height, None, x, y, self.fill_opacity, None, None, **kwargs)


    def _setup_scene(self, scene: Scene) -> None:
        if self.moveToBackground:
            scene.bring_to_back(self.cursor)
        
        super()._setup_scene(scene)
    

class CursorResizeDefault(ApplyMethod):
    def __init__(self, cursor: AltCursor, run_time, **kwargs):
        super().__init__(cursor.changeForm, cursor.DEFAULT_SIZE, cursor.DEFAULT_SIZE, cursor.DEFAULT_SIZE/2.0, None, None, cursor.DEFAULT_CURSOR_OPACITY, None, None, run_time=run_time, **kwargs)
        self.cursor = cursor

    def _setup_scene(self, scene: Scene) -> None:
        scene.bring_to_front(self.cursor) # to front always!
        super()._setup_scene(scene)


class CursorPositionTracking(Animation):
    def __init__(self, cursor, pos: Callable[[], list], **kwargs):
        super().__init__(cursor, **kwargs)
        self.cursor = cursor
        self.pos = pos

    def interpolate_mobject(self, alpha):
        currentPos = self.pos()
        self.cursor.changeForm(x=currentPos[0], y=currentPos[1])


class SophiaCursorScene(SophiaScene):
    def playInBackground(self, animation: Animation, run_time=None):
        if run_time is not None:
            animation = animation.set_run_time(run_time)
        turn_animation_into_updater(animation)

def format_text_group(texts, color = c1t, buff=0.4, buff_inner = 0.1, font_size=fs4):
    texts_formatted = []

    for text in texts:
        if isinstance(text, list):
            # If the element is a sublist, format its elements with low distance
            sublist = [Tex(sub_text, font_size=font_size, color=color) for sub_text in text]
            sublist_group = Group(*sublist).arrange(DOWN, aligned_edge=LEFT, buff=buff_inner)
            texts_formatted.append(sublist_group)
        else:
            # If the element is a string, format it with medium distance
            string = Tex(text, font_size=font_size, color=color)
            texts_formatted.append(string)

    # Group all formatted elements and arrange them with medium distance
    texts_group = Group(*texts_formatted).arrange(DOWN, aligned_edge=LEFT, buff=buff)

    return texts_group

class Notepad(Group):


    def __init__(self,
                 texts = None,
                 buff = 0.4,
                 buff_inner = 0.1,
                 primary_color = c1t,
                 secondary_color = c2,
                 *mobjects,
                 **kwargs):
        super().__init__(*mobjects, **kwargs)
        self.notepad = ImageMobject(assets_folder / "img" / "paper.png").scale(0.95).move_to([0,-2.4,0])
        self.texts = texts
        self.buff = buff
        self.buff_inner = buff_inner
        self.primary_color = Text("").set_color(primary_color).get_color(),
        self.secondary_color = Text("").set_color(secondary_color).get_color()
        self.add(self.notepad)
        
        if self.texts is not None:
            self.texts = format_text_group(self.texts, buff=self.buff).move_to(self.notepad)
            self.add(self.texts)

    def switch_texts(self, texts):
        self.remove(self.texts)
        self.texts = format_text_group(texts, buff=self.buff).move_to(self.notepad)
        self.add(self.texts)

    def change_colors(self, idcs, colors=None):
        if colors is None:
            for idx in idcs:
                if isinstance(self.texts[idx], Group):
                    for subidx in range(len(self.texts[idx])):
                        self.texts[idx][subidx].set_color(self.primary_color if self.texts[idx][subidx].get_color()==self.secondary_color else self.secondary_color)                   
                else:
                    self.texts[idx].set_color(self.primary_color if self.texts[idx].get_color()==self.secondary_color else self.secondary_color)
        else:       
            for idx in idcs:
                self.texts[idx].set_color(colors[idx])


class Bubble(VGroup):

    def __init__(self, texts, width=2, height=1, corner_radius=1/4, center=ORIGIN, start_point = [0,-1.2,0], run_time=2, color=BLACK, loc = "b1"):
        super().__init__()

        #set up the text
        self.text = VGroup(*[Tex(text, color=c1t, font_size=fs3) for text in texts]).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to(center)
        self.text.scale(min(width/self.text.get_width(), height/self.text.get_height())*0.8)

        #Preparations for rectangle and line
        loc = "l2" if "r2" in loc else "r2" if "l2" in loc else loc
        switch_hw = True if 'b' in loc or 't' in loc else False
        if switch_hw:
            width, height = height, width
        rotate = 90 if 't' in loc else 180 if 'l' in loc else 270 if 'b' in loc else 0
        flip = True if '2' in loc else False

        #set up the rectangle and line
        self.rect = RoundedRectangle(width=width, height=height, corner_radius=1/4, color=BLACK).rotate(rotate*DEGREES).shift(center)
        if flip:
            self.rect.flip()
        self.line = Line(start_point, self.rect.get_start(), color=color)
        self.bubble = VGroup(self.line, self.rect)
        self.add(self.line, self.rect)

    def get_end(self):
        return self.line.get_end() if not np.linalg.norm(self.rect.get_end()-self.rect.get_start())>0.00001 else self.rect.get_end()
    
    def get_center(self):
        return self.rect.get_center()
    
class Updown_Arrow(Line):


    ### TODO: Make it work with the tip
    def __init__(self, origin_loc=ORIGIN, vert=True, **kwargs):
        perturb = [0,0.0000001,0]
        super().__init__(start=origin_loc+perturb, end=origin_loc, **kwargs)
        self.vert = vert
        # self.add_tip()
        # tip = self.get_tip()
        self.origin_loc = origin_loc
        self.add_updater(self.update_color)
        # self.add_updater(self.update_tip)

    def update_color(self, dt, arrow):
        if self.vert:
            if self.get_y()>self.origin_loc[1]:
                self.set_color(GREEN)
            else:
                self.set_color(RED)
        else:
            if self.get_x()<self.origin_loc[0]:
                self.set_color(GREEN)
            else:
                self.set_color(RED)
        # print(self.get_length())

    def update_tip(self, dt, arrow):
        if self.get_length()<0.05:
            self.tip.length=self.get_length()
            self.tip.move_to(self.get_end())
        else:
            self.tip.length=0.05
            self.tip.move_to(self.get_end())




# The folder where the assets are stored, include here for compatibility with the old version
assets_folder = SCENES_ASSETS_FOLDER