package main

import (
	"container/heap"         // Для реализации приоритетной очереди
	"encoding/binary"        // Для чтения бинарных данных
	"flag"                   // Для парсинга аргументов командной строки
	"fmt"                    // Для форматированного вывода
	"github.com/fogleman/gg" // Для удобного рисования
	"image/color"            // Для работы с цветами
	"math"                   // Для математических функций
	"os"                     // Для работы с файловой системой
	"time"                   // Для измерения времени выполнения
)

// Point представляет координаты точки на карте
type Point struct {
	X, Y int // Координаты по осям X и Y
}

// Node представляет узел в графе, используемый в алгоритмах поиска
type Node struct {
	Point            // Встраиваемая структура Point для хранения координат
	Cost     float64 // Стоимость пути до этого узла
	Priority float64 // Приоритет узла в приоритетной очереди (для A*)
	Index    int     // Индекс узла в приоритетной очереди (необходим для heap.Interface)
}

// PriorityQueue реализует приоритетную очередь для узлов
type PriorityQueue []*Node

// Len возвращает количество элементов в очереди
func (pq PriorityQueue) Len() int { return len(pq) }

// Less определяет порядок элементов в очереди (меньшее значение Priority имеет более высокий приоритет)
func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Priority < pq[j].Priority
}

// Swap меняет местами два элемента в очереди и обновляет их индексы
func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

// Push добавляет элемент в очередь
func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)           // Текущая длина очереди
	node := x.(*Node)       // Приведение типа интерфейса к *Node
	node.Index = n          // Устанавливаем индекс нового узла
	*pq = append(*pq, node) // Добавляем узел в конец очереди
}

// Pop удаляет и возвращает элемент с наивысшим приоритетом (наименьшее значение Priority)
func (pq *PriorityQueue) Pop() interface{} {
	old := *pq         // Копируем текущую очередь
	n := len(old)      // Определяем длину очереди
	node := old[n-1]   // Берем последний элемент (heap.Pop достает минимальный элемент)
	old[n-1] = nil     // Удаляем ссылку на узел для предотвращения утечек памяти
	node.Index = -1    // Обнуляем индекс узла
	*pq = old[0 : n-1] // Укорачиваем очередь на один элемент
	return node        // Возвращаем узел
}

// update обновляет стоимость и приоритет узла, затем восстанавливает порядок в очереди
func (pq *PriorityQueue) update(node *Node, cost, priority float64) {
	node.Cost = cost         // Обновляем стоимость пути до узла
	node.Priority = priority // Обновляем приоритет узла
	heap.Fix(pq, node.Index) // Восстанавливаем порядок очереди после изменения приоритета
}

// directions задает возможные направления движения (восемь направлений: вверх, вниз, влево, вправо и диагонали)
var directions = []Point{
	{0, -1},  // Вверх
	{0, 1},   // Вниз
	{-1, 0},  // Влево
	{1, 0},   // Вправо
	{-1, -1}, // Вверх-влево (диагональ)
	{1, -1},  // Вверх-вправо (диагональ)
	{-1, 1},  // Вниз-влево (диагональ)
	{1, 1},   // Вниз-вправо (диагональ)
}

// HeightMap представляет карту высот
type HeightMap struct {
	Width, Height int       // Ширина и высота карты
	Data          [][]int16 // Двумерный срез для хранения значений высот каждой клетки
}

// readHeightMap читает карту высот из бинарного файла и возвращает структуру HeightMap
func readHeightMap(filename string) (*HeightMap, error) {
	// Открываем файл для чтения
	file, err := os.Open(filename)
	if err != nil {
		return nil, err // Возвращаем ошибку, если файл не открылся
	}
	defer file.Close() // Отложенное закрытие файла после завершения функции

	var width, height int16 // Объявляем переменные для хранения ширины и высоты
	// Читаем ширину карты (первое значение в файле)
	err = binary.Read(file, binary.LittleEndian, &width)
	if err != nil {
		return nil, err // Возвращаем ошибку, если чтение не удалось
	}
	// Читаем высоту карты (второе значение в файле)
	err = binary.Read(file, binary.LittleEndian, &height)
	if err != nil {
		return nil, err // Возвращаем ошибку, если чтение не удалось
	}

	// Инициализируем двумерный срез для хранения данных карты
	data := make([][]int16, height) // Создаем срез длиной height
	for i := range data {
		data[i] = make([]int16, width) // Для каждой строки создаем срез длиной width
	}

	// Читаем значения высот для каждой клетки карты
	for i := 0; i < int(height); i++ {
		for j := 0; j < int(width); j++ {
			err = binary.Read(file, binary.LittleEndian, &data[i][j]) // Читаем высоту для клетки (i, j)
			if err != nil {
				return nil, err // Возвращаем ошибку, если чтение не удалось
			}
		}
	}

	// Возвращаем заполненную карту высот
	return &HeightMap{
		Width:  int(width),
		Height: int(height),
		Data:   data,
	}, nil
}

// inBounds проверяет, находится ли точка p внутри границ карты
func (hm *HeightMap) inBounds(p Point) bool {
	return p.X >= 0 && p.X < hm.Width && p.Y >= 0 && p.Y < hm.Height
}

func (hm *HeightMap) cost(from, to Point) float64 {
	// Вычисляем разницу высот
	heightDiff := math.Abs(float64(hm.Data[to.Y][to.X] - hm.Data[from.Y][from.X]))
	// Увеличиваем влияние разницы высот
	weight := 25.0 // Подберите значение, которое даст желаемый эффект
	return 1.0 + (weight * heightDiff)
}

// dijkstra реализует алгоритм Дейкстры для поиска кратчайшего пути
// Возвращает стоимость пути, процент просмотренных вершин, время выполнения и сам путь
func dijkstra(hm *HeightMap, start, goal Point) (float64, int, time.Duration, []Point) {
	startTime := time.Now() // Засекаем время начала выполнения алгоритма

	// Инициализируем приоритетную очередь и добавляем начальную точку
	frontier := &PriorityQueue{}                                   // Создаем новую приоритетную очередь
	heap.Init(frontier)                                            // Инициализируем очередь как heap
	heap.Push(frontier, &Node{Point: start, Cost: 0, Priority: 0}) // Добавляем стартовый узел с нулевой стоимостью и приоритетом

	cameFrom := make(map[Point]*Point)   // Карта для восстановления пути (откуда пришли)
	costSoFar := make(map[Point]float64) // Карта для хранения минимальной стоимости до каждой точки
	costSoFar[start] = 0                 // Стоимость до стартовой точки равна 0

	visited := make(map[Point]bool) // Карта для отслеживания посещенных точек

	// Основной цикл алгоритма Дейкстры
	for frontier.Len() > 0 {
		currentNode := heap.Pop(frontier).(*Node) // Извлекаем узел с наименьшей стоимостью
		current := currentNode.Point              // Получаем координаты текущей точки

		// Если достигли цели, завершаем поиск
		if current == goal {
			break
		}

		// Проверяем, уже посещали эту точку
		if visited[current] {
			continue // Если да, пропускаем
		}

		// Помечаем текущую точку как посещенную
		visited[current] = true

		// Проходим по всем возможным направлениям движения
		for _, dir := range directions {
			next := Point{X: current.X + dir.X, Y: current.Y + dir.Y} // Вычисляем координаты соседней точки
			if !hm.inBounds(next) {                                   // Проверяем, находится ли соседняя точка внутри карты
				continue // Если нет, пропускаем ее
			}
			// Вычисляем новую стоимость до соседней точки
			newCost := costSoFar[current] + hm.cost(current, next)
			// Проверяем, существует ли уже стоимость до этой точки и меньше ли она новой
			if c, ok := costSoFar[next]; !ok || newCost < c {
				costSoFar[next] = newCost  // Обновляем стоимость до точки
				heap.Push(frontier, &Node{ // Добавляем соседнюю точку в очередь с обновленной стоимостью
					Point:    next,
					Cost:     newCost,
					Priority: newCost, // В алгоритме Дейкстры приоритет равен стоимости пути
				})
				cameFrom[next] = &current // Записываем, откуда пришли в эту точку
			}
		}
	}

	// Восстанавливаем путь от цели к старту
	path := reconstructPath(cameFrom, start, goal)
	elapsedTime := time.Since(startTime) // Вычисляем время выполнения алгоритма

	// Вычисляем процент просмотренных вершин
	totalVertices := hm.Width * hm.Height
	percentVisited := 0
	if totalVertices > 0 {
		percentVisited = int((float64(len(visited)) / float64(totalVertices)) * 100)
	}

	// Возвращаем стоимость пути, процент просмотренных вершин, время выполнения и сам путь
	return costSoFar[goal], percentVisited, elapsedTime, path
}

// reconstructPath восстанавливает путь от цели к старту, используя карту cameFrom
func reconstructPath(cameFrom map[Point]*Point, start, goal Point) []Point {
	current := goal          // Начинаем с цели
	path := []Point{current} // Инициализируем путь с цели
	for current != start {   // Пока не достигнем стартовой точки
		previous, ok := cameFrom[current] // Ищем, откуда пришли в текущую точку
		if !ok {                          // Если нет записи о предыдущей точке
			return nil // Путь не найден
		}
		current = *previous                      // Переходим к предыдущей точке
		path = append([]Point{current}, path...) // Добавляем ее в начало пути
	}
	return path // Возвращаем полный путь
}

// Heuristic тип для функций эвристики
type Heuristic func(a, b Point) float64

// astar реализует алгоритм A* с заданной эвристической функцией
// Возвращает стоимость пути, процент просмотренных вершин, время выполнения и сам путь
func astar(hm *HeightMap, start, goal Point, heuristic Heuristic) (float64, int, time.Duration, []Point) {
	startTime := time.Now() // Засекаем время начала выполнения алгоритма

	// Инициализируем приоритетную очередь и добавляем начальную точку
	frontier := &PriorityQueue{} // Создаем новую приоритетную очередь
	heap.Init(frontier)          // Инициализируем очередь как heap
	heap.Push(frontier, &Node{   // Добавляем стартовый узел с нулевой стоимостью и приоритетом, равным эвристике
		Point:    start,
		Cost:     0,
		Priority: heuristic(start, goal),
	})

	cameFrom := make(map[Point]*Point)   // Карта для восстановления пути
	costSoFar := make(map[Point]float64) // Карта для хранения минимальной стоимости до каждой точки
	costSoFar[start] = 0                 // Стоимость до стартовой точки равна 0

	visited := make(map[Point]bool) // Карта для отслеживания посещенных точек

	// Основной цикл алгоритма A*
	for frontier.Len() > 0 {
		currentNode := heap.Pop(frontier).(*Node) // Извлекаем узел с наименьшим приоритетом
		current := currentNode.Point              // Получаем координаты текущей точки

		// Если достигли цели, завершаем поиск
		if current == goal {
			break
		}

		// Проверяем, уже посещали эту точку
		if visited[current] {
			continue // Если да, пропускаем
		}

		// Помечаем текущую точку как посещенную
		visited[current] = true

		// Проходим по всем возможным направлениям движения
		for _, dir := range directions {
			next := Point{X: current.X + dir.X, Y: current.Y + dir.Y} // Вычисляем координаты соседней точки
			if !hm.inBounds(next) {                                   // Проверяем, находится ли соседняя точка внутри карты
				continue // Если нет, пропускаем ее
			}
			// Вычисляем новую стоимость до соседней точки
			newCost := costSoFar[current] + hm.cost(current, next)
			// Проверяем, существует ли уже стоимость до этой точки и меньше ли она новой
			if c, ok := costSoFar[next]; !ok || newCost < c {
				costSoFar[next] = newCost                   // Обновляем стоимость до точки
				priority := newCost + heuristic(next, goal) // Вычисляем приоритет с учетом эвристики
				heap.Push(frontier, &Node{                  // Добавляем соседнюю точку в очередь с обновленной стоимостью и приоритетом
					Point:    next,
					Cost:     newCost,
					Priority: priority,
				})
				cameFrom[next] = &current // Записываем, откуда пришли в эту точку
			}
		}
	}

	// Восстанавливаем путь от цели к старту
	path := reconstructPath(cameFrom, start, goal)
	elapsedTime := time.Since(startTime) // Вычисляем время выполнения алгоритма

	// Вычисляем процент просмотренных вершин
	totalVertices := hm.Width * hm.Height
	percentVisited := 0
	if totalVertices > 0 {
		percentVisited = int((float64(len(visited)) / float64(totalVertices)) * 100)
	}

	// Возвращаем стоимость пути, процент просмотренных вершин, время выполнения и сам путь
	return costSoFar[goal], percentVisited, elapsedTime, path
}

// manhattan реализует Манхэттенскую эвристику (расстояние по сетке)
func manhattan(a, b Point) float64 {
	return math.Abs(float64(a.X-b.X)) + math.Abs(float64(a.Y-b.Y))
}

// euclidean реализует Евклидову эвристику (прямое расстояние)
func euclidean(a, b Point) float64 {
	return math.Hypot(float64(a.X-b.X), float64(a.Y-b.Y))
}

// chebyshev реализует Чебышевскую эвристику (максимум из разниц координат)
func chebyshev(a, b Point) float64 {
	dx := math.Abs(float64(a.X - b.X))
	dy := math.Abs(float64(a.Y - b.Y))
	return math.Max(dx, dy)
}

// writeResults записывает результаты алгоритмов в указанный файл
func writeResults(filename string, results []string) error {
	// Создаем или обрезаем файл для записи
	file, err := os.Create(filename)
	if err != nil {
		return err // Возвращаем ошибку, если файл не удалось создать
	}
	defer file.Close() // Отложенное закрытие файла после завершения функции

	// Записываем каждую строку результатов в файл
	for _, res := range results {
		_, err := file.WriteString(res + "\n") // Записываем строку с переходом на новую строку
		if err != nil {
			return err // Возвращаем ошибку, если запись не удалась
		}
	}
	return nil // Возвращаем nil, если все прошло успешно
}

// formatPath форматирует путь из среза точек в строку вида (x1,y1)->(x2,y2)->...->(xn,yn)
func formatPath(path []Point) string {
	if path == nil {
		return "Путь не найден" // Если путь отсутствует, возвращаем соответствующее сообщение
	}
	pathStr := ""
	for i, p := range path {
		if i != 0 {
			pathStr += " -> " // Добавляем стрелку между точками
		}
		pathStr += fmt.Sprintf("(%d,%d)", p.X, p.Y) // Добавляем координаты точки
	}
	return pathStr // Возвращаем сформированную строку пути
}

// AlgorithmResult хранит результаты выполнения одного алгоритма
type AlgorithmResult struct {
	Name           string        // Название алгоритма
	PathLength     float64       // Длина найденного пути
	PercentVisited int           // Процент просмотренных вершин
	ExecutionTime  time.Duration // Время выполнения алгоритма
	Path           string        // Путь в виде строки
}

// visualizePath создает изображение карты высот с нанесенным путем и сохраняет его в файл
func visualizePath(hm *HeightMap, path []Point, filename string, scale int) error {
	// Определяем размеры изображения с учетом масштаба
	imgWidth := hm.Width * scale
	imgHeight := hm.Height * scale

	// Создаем новый контекст для рисования с заданными размерами
	dc := gg.NewContext(imgWidth, imgHeight)
	dc.SetColor(color.White) // Устанавливаем белый фон
	dc.Clear()               // Очищаем контекст

	// Находим минимальную и максимальную высоты для нормализации
	minHeight := int16(math.MaxInt16)
	maxHeight := int16(math.MinInt16)
	for _, row := range hm.Data {
		for _, h := range row {
			if h < minHeight {
				minHeight = h
			}
			if h > maxHeight {
				maxHeight = h
			}
		}
	}

	// Функция для преобразования высоты в цвет
	heightToColorFunc := func(h int16) color.Color {
		return heightToColor(h, minHeight, maxHeight)
	}

	// Рисуем карту высот
	for y := 0; y < hm.Height; y++ {
		for x := 0; x < hm.Width; x++ {
			c := heightToColorFunc(hm.Data[y][x])                                                // Получаем цвет для текущей клетки
			dc.SetColor(c)                                                                       // Устанавливаем цвет
			dc.DrawRectangle(float64(x*scale), float64(y*scale), float64(scale), float64(scale)) // Рисуем прямоугольник
			dc.Fill()                                                                            // Заполняем прямоугольник цветом
		}
	}

	// Рисуем путь, если он существует
	if path != nil && len(path) > 1 {
		dc.SetColor(color.RGBA{0, 0, 0, 255}) // Устанавливаем черный цвет для пути
		dc.SetLineWidth(float64(scale))       // Устанавливаем толщину линии

		// Перемещаемся к первой точке пути
		first := path[0]
		dc.MoveTo(float64(first.X*scale+scale/2), float64(first.Y*scale+scale/2))

		// Рисуем линию к каждой следующей точке пути
		for _, p := range path[1:] {
			dc.LineTo(float64(p.X*scale+scale/2), float64(p.Y*scale+scale/2))
		}

		dc.Stroke() // Отрисовываем линию
	}

	// Отмечаем начало пути зеленым кружком и конец синим
	if path != nil && len(path) > 0 {
		// Начало пути
		start := path[0]
		dc.SetColor(color.RGBA{0, 255, 0, 255}) // Зеленый цвет
		dc.DrawCircle(float64(start.X*scale+scale/2), float64(start.Y*scale+scale/2), float64(scale)/2)
		dc.Fill()

		// Конец пути
		end := path[len(path)-1]
		dc.SetColor(color.RGBA{0, 0, 255, 255}) // Синий цвет
		dc.DrawCircle(float64(end.X*scale+scale/2), float64(end.Y*scale+scale/2), float64(scale)/2)
		dc.Fill()
	}

	// Сохраняем изображение в файл
	return dc.SavePNG(filename)
}

// heightToColor преобразует высоту в цвет, используя градиент от зелёного до красного
func heightToColor(h int16, minHeight, maxHeight int16) color.Color {
	if maxHeight == minHeight {
		return color.RGBA{0, 0, 0, 255} // Черный, если все высоты одинаковы
	}

	normalized := float64(h-minHeight) / float64(maxHeight-minHeight)
	var r, g, b uint8

	switch {
	case normalized < 0.25:
		// Переход от зелёного (0, 255, 0) к жёлтому (255, 255, 0)
		t := normalized / 0.25
		r = uint8(t * 255)
		g = 255
		b = 0
	case normalized < 0.5:
		// Переход от жёлтого (255, 255, 0) к оранжевому (255, 165, 0)
		t := (normalized - 0.25) / 0.25
		r = 255
		g = uint8(255 - t*90) // 255 -> 165
		b = 0
	case normalized < 0.75:
		// Переход от оранжевого (255, 165, 0) к красному (255, 0, 0)
		t := (normalized - 0.5) / 0.25
		r = 255
		g = uint8(165 - t*165) // 165 -> 0
		b = 0
	default:
		// Переход от красного (255, 0, 0) к тёмно-красному (139, 0, 0)
		t := (normalized - 0.75) / 0.25
		r = 255 - uint8(t*116) // 255 -> 139
		g = 0
		b = 0
	}

	return color.RGBA{r, g, b, 255}
}

func main() {
	// Объявляем переменные для параметров командной строки
	inputFile := flag.String("i", "", "Имя входного файла (обязательно)")
	outputFile := flag.String("o", "output.txt", "Имя выходного файла (по умолчанию output.txt)")
	xStart := flag.Int("sx", 0, "Координата X начала пути (обязательно)")
	yStart := flag.Int("sy", 0, "Координата Y начала пути (обязательно)")
	xEnd := flag.Int("ex", 0, "Координата X конца пути (обязательно)")
	yEnd := flag.Int("ey", 0, "Координата Y конца пути (обязательно)")
	scale := flag.Int("scale", 5, "Масштаб изображения (количество пикселей на клетку)")

	flag.Parse() // Парсим аргументы командной строки

	// Проверяем, что обязательный параметр -i (входной файл) указан
	if *inputFile == "" {
		fmt.Println("Ошибка: необходимо указать имя входного файла с помощью параметра -i")
		flag.Usage() // Выводим описание использования программы
		return       // Завершаем программу
	}

	// Читаем карту высот из входного файла
	heightMap, err := readHeightMap(*inputFile)
	if err != nil {
		fmt.Printf("Ошибка при чтении файла: %v\n", err) // Выводим сообщение об ошибке
		return                                           // Завершаем программу
	}

	// Создаем точки начала и конца пути на основе переданных координат
	start := Point{X: *xStart, Y: *yStart}
	goal := Point{X: *xEnd, Y: *yEnd}

	// Проверяем, находятся ли точки начала и конца внутри границ карты
	if !heightMap.inBounds(start) || !heightMap.inBounds(goal) {
		fmt.Println("Ошибка: координаты начала или конца пути выходят за пределы карты")
		return // Завершаем программу
	}

	// Инициализируем срез для хранения результатов работы алгоритмов
	results := []AlgorithmResult{}

	// Запускаем алгоритм Дейкстры
	fmt.Println("Запуск алгоритма Дейкстры...")
	costDijkstra, percentDijkstra, elapsedDijkstra, pathDijkstra := dijkstra(heightMap, start, goal) // Выполняем поиск
	// Формируем строку пути в читаемом формате
	pathStrDijkstra := formatPath(pathDijkstra)
	// Добавляем результаты в срез results
	results = append(results, AlgorithmResult{
		Name:           "Дейкстра",
		PathLength:     costDijkstra,
		PercentVisited: percentDijkstra,
		ExecutionTime:  elapsedDijkstra,
		Path:           pathStrDijkstra,
	})

	// Запускаем алгоритм A* с Манхэттенской эвристикой
	fmt.Println("Запуск алгоритма A* с Манхэттенской эвристикой...")
	costAStarManhattan, percentAStarManhattan, elapsedAStarManhattan, pathAStarManhattan := astar(heightMap, start, goal, manhattan) // Выполняем поиск
	// Формируем строку пути в читаемом формате
	pathStrAStarManhattan := formatPath(pathAStarManhattan)
	// Добавляем результаты в срез results
	results = append(results, AlgorithmResult{
		Name:           "A* (Манхэттенская эвристика)",
		PathLength:     costAStarManhattan,
		PercentVisited: percentAStarManhattan,
		ExecutionTime:  elapsedAStarManhattan,
		Path:           pathStrAStarManhattan,
	})

	// Запускаем алгоритм A* с Евклидовой эвристикой
	fmt.Println("Запуск алгоритма A* с Евклидовой эвристикой...")
	costAStarEuclidean, percentAStarEuclidean, elapsedAStarEuclidean, pathAStarEuclidean := astar(heightMap, start, goal, euclidean) // Выполняем поиск
	// Формируем строку пути в читаемом формате
	pathStrAStarEuclidean := formatPath(pathAStarEuclidean)
	// Добавляем результаты в срез results
	results = append(results, AlgorithmResult{
		Name:           "A* (Евклидова эвристика)",
		PathLength:     costAStarEuclidean,
		PercentVisited: percentAStarEuclidean,
		ExecutionTime:  elapsedAStarEuclidean,
		Path:           pathStrAStarEuclidean,
	})

	// Запускаем алгоритм A* с Чебышевской эвристикой
	fmt.Println("Запуск алгоритма A* с Чебышевской эвристикой...")
	costAStarChebyshev, percentAStarChebyshev, elapsedAStarChebyshev, pathAStarChebyshev := astar(heightMap, start, goal, chebyshev) // Выполняем поиск
	// Формируем строку пути в читаемом формате
	pathStrAStarChebyshev := formatPath(pathAStarChebyshev)
	// Добавляем результаты в срез results
	results = append(results, AlgorithmResult{
		Name:           "A* (Чебышевская эвристика)",
		PathLength:     costAStarChebyshev,
		PercentVisited: percentAStarChebyshev,
		ExecutionTime:  elapsedAStarChebyshev,
		Path:           pathStrAStarChebyshev,
	})

	// Инициализируем срез строк для записи в файл
	outputLines := []string{}
	for _, res := range results {
		// Форматируем время выполнения в миллисекундах
		execTime := fmt.Sprintf("%.2f ms", float64(res.ExecutionTime)/float64(time.Millisecond))
		// Добавляем информацию о текущем алгоритме
		outputLines = append(outputLines, fmt.Sprintf("Алгоритм: %s", res.Name))
		outputLines = append(outputLines, fmt.Sprintf("Длина пути: %.2f", res.PathLength))
		outputLines = append(outputLines, fmt.Sprintf("Процент просмотренных вершин: %d%%", res.PercentVisited))
		outputLines = append(outputLines, fmt.Sprintf("Время выполнения: %s", execTime))
		outputLines = append(outputLines, fmt.Sprintf("Путь: %s", res.Path))
		outputLines = append(outputLines, "") // Добавляем пустую строку для разделения
	}

	// Записываем все результаты в выходной файл
	err = writeResults(*outputFile, outputLines)
	if err != nil {
		fmt.Printf("Ошибка при записи результатов: %v\n", err) // Выводим сообщение об ошибке
		return                                                 // Завершаем программу
	}

	// Визуализируем пути для всех алгоритмов
	fmt.Println("Создание визуализаций путей...")
	// Визуализация пути, найденного Дейкстрой
	err = visualizePath(heightMap, pathDijkstra, "dijkstra_path.png", *scale)
	if err != nil {
		fmt.Printf("Ошибка при создании визуализации Дейкстры: %v\n", err)
		return
	}

	// Визуализация пути, найденного A* с Манхэттенской эвристикой
	err = visualizePath(heightMap, pathAStarManhattan, "astar_manhattan_path.png", *scale)
	if err != nil {
		fmt.Printf("Ошибка при создании визуализации A* (Манхэттенская): %v\n", err)
		return
	}

	// Визуализация пути, найденного A* с Евклидовой эвристикой
	err = visualizePath(heightMap, pathAStarEuclidean, "astar_euclidean_path.png", *scale)
	if err != nil {
		fmt.Printf("Ошибка при создании визуализации A* (Евклидова): %v\n", err)
		return
	}

	// Визуализация пути, найденного A* с Чебышевской эвристикой
	err = visualizePath(heightMap, pathAStarChebyshev, "astar_chebyshev_path.png", *scale)
	if err != nil {
		fmt.Printf("Ошибка при создании визуализации A* (Чебышевская): %v\n", err)
		return
	}

	// Выводим сообщение о завершении записи и создании визуализаций
	fmt.Println("Результаты записаны в файл", *outputFile)
	fmt.Println("Визуализация путей сохранена как:")
	fmt.Println("- dijkstra_path.png")
	fmt.Println("- astar_manhattan_path.png")
	fmt.Println("- astar_euclidean_path.png")
	fmt.Println("- astar_chebyshev_path.png")
}
