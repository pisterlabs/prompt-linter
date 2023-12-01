from openaid.projects.models import Project, Initiative


def run():

    # update all project initiatve relation
    for p in Project.objects.exclude(number=''):
        code = p.number.split('/')[0].zfill(6)

        try:
            i = Initiative.objects.get(code=code)
            p.initiative = i
            p.save()
            print 'Add initiative %s to project %s' % (i, p)
        except Initiative.DoesNotExist:
            print 'Initiative does not exists [%s]' % p.number

    for i in Initiative.objects.all():
        i.total_project_costs = i._get_first_project_value('total_project_costs')
        i.grant_amount_approved = sum(i._project_fields_map('grant_amount_approved',skip_none=True))
        i.loan_amount_approved = sum(i._project_fields_map('loan_amount_approved',skip_none=True))
        i.save()

        print 'Initiative %s [total:%s | grant:%s | loan:%s]' % (
            i, i.total_project_costs, i.grant_amount_approved, i.loan_amount_approved
        )

