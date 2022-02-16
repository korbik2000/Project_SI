# Project_SI
Opis:
1. Tworzymy wzorzec do wczytywania zdjęć, zachodzi w nim: zmiana rozmiaru, konwersja do tensora i normalizacja.
2. Wczytujemy dane treningowe.
3. Uczymy model za pomocą funkcji fit z bilbioteki detecto.
4. W przypadku braku konieczności przeprowadzenia uczenia, można wczytać model (odkomentować oznaczoną linijkę i zakomentować uczenie)
5. W pętli sprawdzamy każdy obrazek w formacie .png z folderu 'Test/images':
 - za pomocą funkcji predict z bilbioteki detecto otrzymujemy: podpis wykrytego obiektu, pozycję ramki oraz wynik punktowy określający pewność,
 - filtrujemy wyniki, żeby pozostały tylko rezulaty z najwyższymi punktami (>0.75)
 - sprawdzamy które z wyników są znakami przejść dla pieszych, zapisujemy ich liczbę i pozycję ramki,
 - sprawdzamy czy znak przejścia zajmuje 1/10 szerokości i wyskości zdjęcia,
 - wypisujemy printem nazwę pliku, liczbę znaków przejść dla pieszych na danym obrazku i pozycję ramki (xmin xmax ymin ymax).

Uwagi:
-plik wyuczonego modelu zajmuje ponad 150MB i nie da się go umieścić na githubie
-do wytrenowania modelu użyłem narzędzia google collab z akceleracją GPU
