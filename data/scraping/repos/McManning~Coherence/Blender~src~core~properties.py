
import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
    CollectionProperty
)

from . import runtime, scene_objects, image_sync
from util.registry import autoregister

def on_toggle_component_enabled(self, context):
    """Callback for when the enabled property changes"""
    component = self.get_component()
    if component:
        component.enabled = self.enabled

@autoregister
class CoherenceSceneProperties(PropertyGroup):
    """The primary Coherence connection settings"""

    #: str, default ``Coherence``: Shared connection name that matches with Unity's Coherence Settings window
    connection_name: StringProperty(
        name='Buffer Name',
        default='Coherence',
        description='This name must match the buffer name in Unity\'s Coherence Settings window'
    )

    #: float, default 0.05: How frequently (in seconds) to sync image pixel data while actively using the Image Paint tool
    texture_slot_update_frequency: FloatProperty(
        name='Image Update Frequency',
        description='How frequently (in seconds) to sync image pixel data while actively using the Image Paint tool',
        default=0.05
    )

    #: bool, default True: Show the Coherence toggle button in the viewport controls menu
    show_view3d_controls: BoolProperty(
        name='Show Viewport Controls',
        description='Show the Coherence toggle button in the viewport controls menu',
        default=True
    )

    @classmethod
    def register(cls):
        bpy.types.Scene.coherence = PointerProperty(
            name='Coherence Renderer Settings',
            description='',
            type=cls
        )

    @classmethod
    def unregister(cls):
        del bpy.types.Scene.coherence

@autoregister
class CoherenceComponentMetadata(bpy.types.PropertyGroup):
    """Metadata for a Coherence Component currently attached to an object

    This is stored in a CollectionProperty and persisted with the object
    so we can restore component states when loading Coherence.
    """
    enabled: BoolProperty(
        name = 'Toggle enabled',
        description = 'This will also toggle the linked Unity component',
        update=on_toggle_component_enabled
    )

    is_builtin: BoolProperty(default=True)
    expanded: BoolProperty(default=False)
    guid: StringProperty(default='')

    def get_component(self):
        """Retrieve the component instance for this metadata properties.

        Returns:
            Component|None
        """
        plugin = runtime.instance.get_plugin(scene_objects.SceneObjects)
        return plugin.get_component_by_guid(self.id_data, self.guid)

@autoregister
class CoherenceObjectProperties(PropertyGroup):
    components: CollectionProperty(
        type=CoherenceComponentMetadata
    )

    @classmethod
    def register(cls):
        bpy.types.Object.coherence = PointerProperty(
            name='Coherence Settings',
            description='',
            type=cls
        )

    @classmethod
    def unregister(cls):
        del bpy.types.Object.coherence


# TODO: Move to image.py
def validate_image_for_sync(img) -> str:
    """Check if an image can be synced with Unity

    Args:
        img (bpy.types.Image)

    Returns:
        str: Error message, or an empty string for no error
    """
    w, h = img.size

    # Perform additional checks for image format - ensuring we can transfer
    # it in RGBA32 without significant conversion overhead
    #if img.depth != 32:
    #    return 'Image must contain an alpha channel'

    # TODO: Server should provide what the limits are.
    if w < 1 or h < 1 or w > 1024 or h > 1024:
        return 'Image must be between 1x1 and 1024x1024 to enable syncing'

    return ''

def texture_slot_enum_items(self, context):
    """Generate an `EnumProperty` items list from Coherence texture slots

    Args:
        context (:mod:`bpy.context`)

    Returns:
        list of [(slot, slot, ''), ...]
    """
    slots = image_sync.get_texture_slots()
    return [(name, name, '') for name in slots]

def on_toggle_image_enabled(self, context):
    """Callback for when the enabled property changes"""
    print('Image toggled to {}'.format(self.enabled))
    # TODO: Remove this? It's automatically detected now.

@autoregister
class CoherenceImageProperties(PropertyGroup):
    #: str, default None: An error with the source image that prevents syncing
    error: StringProperty(
        name='Source Image Error',
        description='An error with the source image that prevents syncing',
        default='',
        # Errors are runtime only
        options={'SKIP_SAVE'}
    )

    format: IntProperty(
        name='Format',
        description='',
        default=-1,
        options={'SKIP_SAVE'}
    )

    enabled: BoolProperty(
        name = 'Enable sync',
        description = '',
        update=on_toggle_image_enabled
    )

    #: str, default None: The named image slot to sync pixels with
    # texture_slot: EnumProperty(
    #     name='Slot',
    #     description='The named image slot to sync pixels with',
    #     items=texture_slot_enum_items,
    #     update=_on_update_texture_slot,
    #     default=0,
    #     # Don't persist slot targets between Blender saves,
    #     # as these may be modified within Unity.
    #     options={'SKIP_SAVE'}
    # )

    @classmethod
    def register(cls):
        bpy.types.Image.coherence = PointerProperty(
            name='Coherence Image Settings',
            description='',
            type=cls
        )

    @classmethod
    def unregister(cls):
        del bpy.types.Image.coherence
