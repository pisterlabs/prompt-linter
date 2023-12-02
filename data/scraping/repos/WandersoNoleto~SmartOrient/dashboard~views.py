from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from guidances.forms import GuidanceRegisterForm
from guidances.models import Guidance
from users.decorators import user_has_permission
from users.models import Advisor, Coordination, Student


@login_required(login_url= '/auth/login/')
@user_has_permission(allowed_roles=['Students'])
def home(request):
    guidances      = Guidance.objects.filter(status="Em andamento", student_id=request.user.id)
    logged_user_id = request.user.id
    logged_user    = Student.objects.filter(genericuser_ptr_id=logged_user_id).first()
    formGuidance   = GuidanceRegisterForm

    context = {
        'guidances': guidances,
        'logged_user': logged_user,
        'formGuidance': formGuidance
        }
    
    return render(request, 'studentHome.html', context)

@login_required(login_url= '/auth/login/')
@user_has_permission(allowed_roles=['Advisors'])
def home_advisor(request):
    guidances        = Guidance.objects.filter(status="Em andamento", advisor_id=request.user.id)
    pending_guidances = Guidance.objects.filter(status="Pendente", advisor_id=request.user.id)
    logged_user_id = request.user.id
    logged_user    = Advisor.objects.filter(genericuser_ptr_id=logged_user_id).first()
    formGuidance   = GuidanceRegisterForm

    context = {
        'guidances': guidances,
        'logged_user': logged_user,
        'formGuidance': formGuidance,
        'pending_guidances': pending_guidances
        }
    
    return render(request, 'advisorHome.html', context)

@login_required(login_url= '/auth/login/')
@user_has_permission(allowed_roles=['Coordinations'])
def home_coordination(request):
    guidances      = Guidance.objects.filter(status="Em andamento", coordination_id=request.user.id)
    logged_user_id = request.user.id
    logged_user    = Coordination.objects.filter(genericuser_ptr_id=logged_user_id).first()
    print(logged_user.course)
    context = {
        'guidances': guidances,
        'logged_user': logged_user,
        }
    
    return render(request, 'coordinationHome.html', context)

