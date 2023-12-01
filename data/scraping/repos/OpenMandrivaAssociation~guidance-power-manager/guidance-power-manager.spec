Name:           guidance-power-manager
Summary:        KDE power management applet
Version:        4.4.0
Release:        4
Url:            http://websvn.kde.org/trunk/extragear/utils/guidance-power-manager
License:        GPLv2+
Group:          Graphical desktop/KDE
Source0:        http://fr2.rpmfind.net/linux/KDE/stable/%version/src/extragear/%{name}-%{version}.tar.bz2
BuildRequires:  pkgconfig(xscrnsaver)
BuildRequires:	pkgconfig(xrandr)
BuildRequires:	pkgconfig(xxf86vm)
BuildRequires:	pkgconfig(x11)
BuildRequires:  python-devel
BuildRequires:  python-sip
BuildRequires:  python-dbus
BuildRequires:  python-kde4
BuildRequires:	kdelibs4-devel

Requires:       pm-utils
Requires:       kdebase4-runtime
Requires:       python-kde4
Requires:	    python-dbus

%description
The package provides battery monitoring and suspend/standby triggers.
It is based on the powersave package and therefore supports APM and
ACPI. See powersave package for additional features such as CPU frequency
scaling(SpeedStep and PowerNow) and more

%files -f %name.lang
%defattr(-,root,root) 
%_kde_bindir/guidance-power-manager
%py_platsitedir/ixf86misc.so
%py_platsitedir/xf86misc.py
%_kde_appsdir/guidance-power-manager
%_kde_datadir/autostart/guidance-power-manager.desktop

#--------------------------------------------------------------------

%prep
%setup -q -n %name-%version

%build
%cmake_kde4
%make

%install
rm -fr %buildroot
%makeinstall_std -C build

%find_lang %name

%clean
rm -fr %buildroot


%changelog
* Wed Nov 17 2010 Funda Wang <fwang@mandriva.org> 4.4.0-2mdv2011.0
+ Revision: 598142
- fix BR
- add linkage fix

  + Bogdano Arendartchuk <bogdano@mandriva.com>
    - rebuild for python 2.7

* Sat Feb 13 2010 Funda Wang <fwang@mandriva.org> 4.4.0-1mdv2010.1
+ Revision: 505237
- BR libxrandr-devel
- new version 4.4.0

* Tue Sep 01 2009 Nicolas Lécureuil <nlecureuil@mandriva.com> 4.3.1-1mdv2010.0
+ Revision: 423837
- Update to version 4.3.1

* Thu Jan 29 2009 Funda Wang <fwang@mandriva.org> 4.2.0-1mdv2009.1
+ Revision: 335339
- New version 4.2.0

* Tue Jan 20 2009 Nicolas Lécureuil <nlecureuil@mandriva.com> 4.1.96-1mdv2009.1
+ Revision: 331748
- Update to 4.1.96

* Mon Sep 08 2008 Funda Wang <fwang@mandriva.org> 4.1.1-2mdv2009.0
+ Revision: 282550
- fix requires (bug#43206)

* Thu Sep 04 2008 Funda Wang <fwang@mandriva.org> 4.1.1-1mdv2009.0
+ Revision: 280231
- New version 4.1.1

* Thu Aug 14 2008 Nicolas Lécureuil <nlecureuil@mandriva.com> 4.1.0-1mdv2009.0
+ Revision: 272155
- Add buildrequires
- Add python-sip as Buildrequire
- Add python-devel as buildrequire
- import guidance-power-manager


