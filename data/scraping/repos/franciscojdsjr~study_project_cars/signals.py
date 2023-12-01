from django.db.models.signals import pre_delete, pre_save, post_save, post_delete
from django.db.models import Sum
from django.dispatch import receiver
from cars.models import Car, CarIventory
from openai_api.client import get_car_ai_bio


def car_inventory_update():
    cars_count = Car.objects.all().count()
    cars_value = Car.objects.aggregate(
        total_value=Sum('value')
    )['total_value']
    CarIventory.objects.create(
        cars_count=cars_count,
        cars_value=cars_value
    )


@receiver(pre_save, sender=Car)
def car_pre_save(sender, instance, **kargs):
    if not instance.bio:
        ai_bio = get_car_ai_bio(
            instance.model, instance.brand, instance.model_year
        )
        instance.bio = ai_bio


@receiver(post_save, sender=Car)
def car_post_save(sender, instance, **kargs):
    car_inventory_update()


@receiver(pre_delete, sender=Car)
def car_pre_delete(sender, instance, **kargs):
    print('### PRE DELETE ###')


@receiver(post_delete, sender=Car)
def car_post_delete(sender, instance, **kargs):
    car_inventory_update()
