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

	// Добавляем кнопку Force Move
	forceMove := widget.NewButtonWithIcon("Force Move", theme.MediaPlayIcon(), func() {
		board.ForceNextMove()
	})

	// Добавляем кнопку Force Move Neuro
	forceMoveNeuro := widget.NewButtonWithIcon("Force Move Neuro", theme.MediaPlayIcon(), func() {
		board.ForceNextMoveNeuro()
	})

    // Добавляем кнопку отмены хода (Take Back)
	takeBack := widget.NewButtonWithIcon("Take Back", theme.ContentUndoIcon(), func() {
		for _, obj := range grid.Objects {
			stack := obj.(*fyne.Container)
			icon := stack.Objects[1].(*boardIcon)
			icon.Reset()
		}
		board.TakeBack()
	})

	// Объединяем все три кнопки в один ряд
	buttons := container.NewHBox(reset, forceMove, takeBack, forceMoveNeuro)

	board.Reset(true)

	go syncPeriodic(board, sd)

	// Передаем контейнер с кнопками в верхнюю часть (top)
	return container.NewBorder(buttons, nil, nil, nil, grid)
}
