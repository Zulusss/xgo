package xfyneui

import (
	// "image/color"

	"fyne.io/fyne/v2"
	// "fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"github.com/ratamahata/xgo/xai"
)

// Show loads a tic-tac-toe example window for the specified app context
func Show(win fyne.Window, sd *StatusController) fyne.CanvasObject {
	gb := xai.GetXBoard(0)
	go gb.Grow()

	board := &board{gb: gb}

	// 1. Определяем мапу соответствия Код -> Название
	modeToName := map[string]string{
		"HH": "Human vs Human",
		"HN": "Human vs Neuro",
		"HL": "Human vs Legacy",
		"TM": "Neuro mixed train",
		"TN": "Neuro self play",
	}

	// 2. Создаем упорядоченный список кодов для отображения в нужном порядке
	codes := []string{"HH", "HN", "HL", "TM", "TN"}

	// Создаем список названий для виджета Select
	var options []string
	for _, code := range codes {
		options = append(options, modeToName[code])
	}

	// 3. Создаем выпадающий список
	modeSelect := widget.NewSelect(options, func(selectedName string) {
		// Ищем код по выбранному названию
		for code, name := range modeToName {
			if name == selectedName {
				board.SwitchPlayMode(code)
				break
			}
		}
	})

	// Устанавливаем значение по умолчанию (например, первый режим)
	modeSelect.SetSelected(modeToName["TM"])

	grid := container.NewGridWithColumns(15)
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			grid.Add(newBoardIcon(r, c, board))
		}
	}

	reset := widget.NewButtonWithIcon("Reset Board", theme.ViewRefreshIcon(), func() {
		for _, obj := range grid.Objects {
			stack := obj.(*fyne.Container)
			icon := stack.Objects[1].(*boardIcon)
			icon.Reset()
		}
		board.Reset(false)
	})

	forceMove := widget.NewButtonWithIcon("Force Move", theme.MediaPlayIcon(), func() {
		board.ForceNextMove()
	})

	forceMoveNeuro := widget.NewButtonWithIcon("Force Move Neuro", theme.MediaPlayIcon(), func() {
		board.ForceNextMoveNeuro()
	})

	takeBack := widget.NewButtonWithIcon("Take Back", theme.ContentUndoIcon(), func() {
		for _, obj := range grid.Objects {
			stack := obj.(*fyne.Container)
			icon := stack.Objects[1].(*boardIcon)
			icon.Reset()
		}
		board.TakeBack()
	})

	buttons := container.NewHBox(reset, forceMove, takeBack, forceMoveNeuro, modeSelect)

	board.Reset(true)
	go syncPeriodic(board, sd)

	// Передаем контейнер с кнопками в верхнюю часть (top)
	return container.NewBorder(buttons, nil, nil, nil, grid)
}
