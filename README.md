## AMfe - Finite Element Research Code at the Chair of Applied Mechanics

   Dieser FE-Forschungscode wird von einem Teil der Numerik-Arbeitsgruppe entwickelt, gewartet und angewendet. Weitere Dokumentationen zu diesem Code sind im Ordner `docs/` zu finden.
   Um die Dokumentation zu bauen, müssen nachfolgende Softwarepakete installiert sein:

   - Python Version 3.4 oder höher
   - Python-Paket sphinx 1.3 oder höher (muss evtl. mit pip3 installiert werden). Version 1.2 funktioniert leider nicht.
   - Python-Paket numpydoc

   Die Dokumentation kann gebaut werden, wenn

      make html

   im Ordner `docs/` ausgeführt wird.
   
### Hinweise:
sphinx muss für pyhton3 installiert sein. Es kann sein, dass sphinx automatisch für python2 installiert wurde. 
Unter python3 kann in der Konsole folgendermaßen geprüft werden, welche sphinx-Version installiert ist:
```python
python3
>>> import sphinx
>>> sphinx.__version__
```
Hier sollte mindestens `'1.3.1` ausgeben werden. 