from django.contrib import admin
from .models import GuidanceArea,Monsters,WeaponsCustom,WeaponsName,WeaponsPartsEffect,WeaponsRare,WeaponsUpgrade

# Register your models here.
admin.site.register(GuidanceArea)
admin.site.register(Monsters)
admin.site.register(WeaponsCustom)
admin.site.register(WeaponsName)
admin.site.register(WeaponsPartsEffect)
admin.site.register(WeaponsRare)
admin.site.register(WeaponsUpgrade)