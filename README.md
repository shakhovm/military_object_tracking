# Відслідковування військових об'єктів
Репозиторій, де міститься програма для відслідковування одиночних об'єктів, а також імплементація стабілізації рамки.

**Встановлення**

Цей репозиторій використовує бібліотеку [Pytracking](https://github.com/visionml/pytracking). Тому інструкції для встановлення середовища варто використати звідти.

Також потрібно виконати наступні команди
```bash
mkdir ltr/external
cd ltr/external
git clone https://github.com/vacancy/PreciseRoIPooling
cd ../..
```
Натреновані моделі та приклади даних можна завантажити [тут](https://drive.google.com/drive/folders/1mLHYTOGFyuE-Y0X2yEH4TImzGmtn_kOh?usp=sharing)

Для того, щоб перетворити відно в директорію з зображеннями потрібно ввести команда

```bash
python labeling_utils/video_to_imgs.py data
```

**Робота програми**

Для запуску програми на відео потрібно ввести наступну команду:
```bash
python tracker.py data/005.mp4 atom
```
Після цього потрібно мишкою обвести об'єкт на натиснути ENTER. Для виходу з програму можна натиснути 'q'

Для оцінки трекера та його роботи разом з фазовою кореляцією та фільтром Калмана потрібно ввести команду :
```
python model_evaluator.py data/005 1 atom
```
data/005 - шлях до послідовності зображень, 1 - включити візуалізацію, atom - архітектура моделі (опції tomp, atom, dimp)

**Важливі файли**

* materials.ipynb - містить аналіз стабілізації траєкторій за допомогою фазової кореляції та фільтру Калмана
* phase_correlation.ipynb - показує приклад роботи фазової кореляції для зміщеного зображення та порівнює результати власної імлпементації та існуючої
* директорія augmentation містить два види аугментації - додавання погодних умов та дестабілізація
* директорія labeling_utils містить важливі файли з кодом для анотування
