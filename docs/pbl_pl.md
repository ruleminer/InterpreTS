# Podsumowanie projektu

## Temat projektu
Projekt opierał się na stworzeniu biblioteki w Python do zadań związanych z ekstracją cech dla szeregów czasowych w celu zastosowania ich do treningu i interpretowania modeli uczenia maszynowego. Stworzono biblioteke:

**interpreTS: Oprogramowanie wspierające interpretowalność i wyjaśnialność modeli predykcyjnych poprzez ekstrakcję cech z szeregów czasowych**

## Członkowie zespołu projektowego
- Martyna Żur
- Sławomir Put
- Weronika Wołowczyk
- Martyna Kramarz
- Jarosław Stzrelczyk
- Piotr Krupiński

## Opiekunowie projektu
- Opiekun głółwny: dr Łukasz Wróbel
- Opiekun pomocniczy:dr Michał Kozielski

## Przyjęte założenia
Projekt koncentrował się na rozwijającym się obszarze sztucznej inteligencji, jakim jest interpretowalność i wyjaśnialność modeli (XAI – Explainable Artificial Intelligence). Tworzenie interpretowalnych modeli na danych złożonych, takich jak szeregi czasowe, jest wyjątkowo trudne.  

Z tego powodu zaplanowano stworzenie narzędzia umożliwiającego użytkownikom generowanie cech opisujących szereg czasowy w sposób bardziej zrozumiały i dostępny. Szczególny nacisk położono na:  
- Generowanie cech łatwych do interpretacji przez użytkowników.
- Zastosowanie technologii obliczeń rozproszonych dla przyspieszenia procesu.  
- Udostępnienie przyjaznego graficznego interfejsu użytkownika.
- Udostępnienie stworzonej biblioteki wraz z dokumentacją na platformie GitHub. 
- Analizę relacji pomiędzy różnymi szeregami czasowymi oraz pomiędzy okresami w tych danych.  

## Osiągnięte cele
Projekt zrealizował szereg kluczowych celów:  

1. Stworzono bibliotekę w języku Python umożliwiającą automatyczną ekstrakcję cech z szeregów czasowych, zaimplementowano 30 cech.  
2. Opracowano mechanizmy pozwalające na uruchamianie obliczeń w środowiskach rozproszonych, za pocą Dash.  
3. Dodano funkcjonalności do interpretowania zaimplementowanych cech, obsłyga streamimgu, presonalizowanie listy cech w Ekstraktrze.
4. Zaprojektowano i wdrożono intuicyjny interfejs graficzny działający w przeglądarce internetowej, umożliwiający przetwarzanie danych bez konieczności pisania kodu, za pomocą Strimly.  
5. Dokonano ewaluacji narzędzia, wykorzystując benchmarkowe zbiory danych do porównania efektywności i jakości w stosunku do istniejących bibliotek, takich jak `tsfresh`.  
6. Udokumentowano i opublikowano wyniki w postaci repozytorium na GitHub oraz pakietu w repozytorium PyPI. Stworzono pełną dokumentację projektu za pomocą Sphinx.

## Zastosowane metody realizacji
Realizacja projektu przebiegała zgodnie z nowoczesnymi standardami w tworzeniu oprogramowania. Wykorzystano:  

- **Technologie**,  zastsowono nowoczesne technologie spierające działania biblioteki takie jak Dask do obliczeń rozporszonyc, Streamly do tworzenia GUI biblioteki czy biblioteki w pythonie do zastosowań uczenia maszynowego.
- **Dokumentacja**,  Stworzono pliki dokumentacji i skonfigurowano je w Sphinx, a także użyto docstringów do opisu funkcji biblioteki, umożliwiając automatyczne generowanie dokumentacji.
- **Pracę zespołową na platformie GitHub**, zapewniającą kontrolę wersji, zgłaszanie problemów, dokumentację oraz wymianę informacji w zespole.  
- **Testy jednostkowe**, za pomocą pytest przetestowano wszystkie funkcjonalności biblioteki.
- **Benchmarkowe testy analityczne**, umożliwiające ocenę jakości wypracowanych rozwiązań oraz ich porównanie z innymi popularnymi bibliotekami.  

## Osiągnięte wyniki
Końcowo stworzono biblioteke interpreTS, zawierająca Ekstraktor Cech dla szeregów czasowych. Biblioteka posiada moduły dla Ekstraktora: kowersji danych, managera danych, walidacji danych, ładowania cech do ekstraktowa oraz managera zadań. W bibliotece zaimplememtowa 30 różnych cech (napisać wszystkie). Dodatkowo przeprowadzono testy jednostkowe, wydajnościowe oraz sparawdzające wykonywanie się cech. Dodano streming o obsługe obliczeń rozporszonych

Biblioteka została porównana z innymi bibliotekami takimi jak 'tsfresh', żeby sparwdzić jej działania i funkcjonalności. Dodatkowo stworzonu use cases w postaci notatników w plikach projektu.

Cała biblioteka została opisana za pomocą Sphinx w dokuemnatcji gdzie znalazły się opis biblioteki, opis Ekstraktora, opis funkcjonalności oraz Tutoriale z przykładami użycia i sposobem działania bibilioteki.

## Inne informacje o projekcie
Podczas realizacji projektów pojawiły się wyzwania związane z realizacją poszczególnych funkcjonalności. Początkowe wersje biblioteki nie były w pełni poprawne i zawierały braki związane z nieznajomością szeregów czasowych w sytraczający sposób oraz całośći biblioteki. Ciężkie okazało się zrozumienie w którą stronę powinno się iść w funkcjonalności i jak dokłądnie biblioteka powinna działać.

Planowane jest napisanie artykułu na temat stworzonego narzędzia. Opisane zostanie czym dokładnie jest, jak zostało stworzonę i co oferuję. Dodatkowo możliwa jest rozbudowa biblioteki o dodatkowe cechy, bardziej specyficzne dla szeregów czasowych oraz dodanie modeli NLP do interpretowalności poszczególnych cech modeli.