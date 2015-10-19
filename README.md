AMfe - Finite Element Research Code at the Chair of Applied Mechanics
---------------------------------------------------------------------

(c) 2015 Lehrstuhl für Angewandte Mechanik, Technische Universität München

Dieser FE-Forschungscode wird von einem Teil der Numerik-Arbeitsgruppe entwickelt, gewartet und angewendet. 


Übersicht: 
----------
1.  [Dokumentation](#1-dokumentation)
2.  [Fortran-Routinen](#2-fortran-routinen)
3.  [Hinweise](#3-hinweise)


1. Dokumentation
----------------
Weitere Dokumentationen zu diesem Code sind im Ordner `docs/` zu finden.
Um die Dokumentation zu bauen, müssen nachfolgende Softwarepakete installiert sein:

   - Python Version 3.4 oder höher
   - Python-Paket sphinx 1.3 oder höher (muss evtl. mit pip3 installiert werden). Version 1.2 funktioniert leider nicht.
   - Python-Paket numpydoc

Die Dokumentation kann gebaut werden, wenn
```bash
make html
```
im Ordner `docs/` ausgeführt wird.
   

2. Fortran-Routinen
-------------------
Um die Fortran-Routinen (für die Assemblierung und die Elemente) zu nutzen, muss das Skript `install_fortran_routines.sh` im Ordner `f2py/` ausgeführt werden.
Hierzu ist es wichtig, dass die Fortran Compiler installiert sind (z.B. `gfortran`, `gfortran-4.8`). 

   
3. Hinweise
-----------

### Sphinx:

`sphinx` muss für `pyhton3` installiert sein. Es kann sein, dass `sphinx` automatisch für `python2` installiert wurde. 
Unter `python3` kann in der Konsole folgendermaßen geprüft werden, welche `sphinx`-Version installiert ist:
```python
python3
>>> import sphinx
>>> sphinx.__version__
```
Hier sollte mindestens `'1.3.1'` ausgegeben werden.


### Spyder:

Empfohlen wir die Entwicklungsumgebung `spyder3` für `python3`.
