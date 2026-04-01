package xfyneui

import (
	"time"

	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"github.com/ratamahata/xgo/xai"
)

type board struct {
	gb                  xai.Grower
	pieces              [15][15]int
	icons               [15][15]*boardIcon
	displayedMovesCount int
	finished            bool
	lastMove            *boardIcon
	lMCoolDown          int
	expectX             bool
	expectO             bool
	humanX              bool //true if Human plays for X
}

func (b *board) newClick(row, column int) {

	xb := b.gb
	xb.GridClick(column, row)
	b.expectO = true
	b.expectX = true
}

func (b *board) Reset(initial bool) {

	actualMoves := b.gb.GetMovesCount()

	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			b.pieces[r][c] = 0
		}
	}

	if !initial && actualMoves <= 1 {

		b.gb.MoveClick()
		b.humanX = true

	} else {
		b.gb.RestartClick()
		b.finished = false
		b.displayedMovesCount = 0
		b.lastMove = nil
		b.lMCoolDown = 0
		b.expectO = false
		b.expectX = false
		b.humanX = false
	}
}

func (b *board) ForceNextMove() {
	b.gb.MoveClick()
}

func (b *board) ForceNextMoveNeuro() {
	b.gb.MoveNeuroClick()
}

func (b *board) TakeBack() {

	b.gb.TakeBackClick()

	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			b.pieces[r][c] = 0
		}
	}

}

type boardIcon struct {
	widget.BaseWidget
	board       *board
	row, column int
	rect        *canvas.Rectangle
	icon        *canvas.Image // Используем напрямую картинку вместо виджета Icon
}

func (i *boardIcon) Tapped(ev *fyne.PointEvent) {
	if i.board.pieces[i.row][i.column] != 0 || i.board.finished {
		return
	}

	i.board.newClick(i.row, i.column)
}


func newBoardIcon(row, column int, board *board) *boardIcon {
	i := &boardIcon{board: board, row: row, column: column}
	i.ExtendBaseWidget(i)
	i.rect = canvas.NewRectangle(color.Transparent)
	i.rect.StrokeWidth = 1
	i.rect.StrokeColor = theme.ForegroundColor()
	i.icon = canvas.NewImageFromResource(nil) // Изначально пусто
	i.icon.FillMode = canvas.ImageFillContain
	board.icons[row][column] = i
	return i
}

// CreateRenderer определяет, как выглядит виджет
func (i *boardIcon) CreateRenderer() fyne.WidgetRenderer {
	content := container.NewStack(i.rect, i.icon)
	return widget.NewSimpleRenderer(content)
}

// Refresh вызывается системой автоматически при смене темы
func (i *boardIcon) Refresh() {
	// 1. Обновляем цвет рамки (черный в светлой / белый в темной)
	i.rect.StrokeColor = theme.ForegroundColor()
	i.rect.Refresh()

	// 2. Обновляем саму иконку (X или O)
	// Fyne сам перекрасит theme.CancelIcon(), если тема изменилась
	i.icon.Refresh()

	// 3. Обновляем весь виджет
	i.BaseWidget.Refresh()
}

func sync(b *board, sd *StatusController) {

	if b.finished {
		return
	}

	xb := b.gb

	actualMoves := xb.GetMovesCount()

	sd.UpdateStatus(0, xb.GetMsg1())
	sd.UpdateStatus(1, xb.GetMsg2())
	sd.UpdateStatus(2, xb.GetMsg3())
	sd.UpdateStatus(3, xb.GetMsg4())
	sd.UpdateStatus(4, xb.GetMsg5())
	sd.UpdateStatus(5, xb.GetMsg6())
	sd.UpdateStatus(6, xb.GetMsg7())
	sd.UpdateStatus(7, xb.GetMsg9())
	sd.UpdateStatus(8, xb.GetMsg8())

	defaultSize := b.icons[0][0].Size()

	if b.lastMove != nil && actualMoves >= 7 { //visual effect for last move

		if b.lastMove.Size().Width < defaultSize.Width {
			b.lastMove.Resize(b.lastMove.Size().AddWidthHeight(defaultSize.Width/10.0, defaultSize.Height/10))
		} else {
			if b.lMCoolDown > 0 {
				b.lMCoolDown--
			} else {
				b.lastMove.Resize(b.lastMove.Size().SubtractWidthHeight(defaultSize.Width/10.0, defaultSize.Height/10))
				b.lMCoolDown = 12
			}
		}
	}

	//if !b.expectX && !b.expectO && actualMoves <= b.displayedMovesCount {
	//	return
	//}

	addedCount := 0
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {

			code := xb.GetCell(r, c)

			if b.pieces[r][c] != code {

				b.pieces[r][c] = code
				lastMove := b.icons[r][c]

                if code > 0 {
                    lastMove.SetIcon(theme.CancelIcon())
                    b.expectX = false
                    addedCount++

                } else if code < 0 {
                    lastMove.SetIcon(theme.RadioButtonIcon())
                    b.expectO = false
                    addedCount++
                } else {
                    lastMove.ResetIcon()
                }

				if b.humanX == (code < 0) {

					if b.lastMove != lastMove {

						if b.lastMove != nil {
							b.lastMove.Resize(defaultSize)
						}
						b.lastMove = lastMove

						if actualMoves >= 7 {
							b.lastMove.Resize(b.lastMove.Size().SubtractWidthHeight(defaultSize.Width*0.2, defaultSize.Height*0.2))
						}
					}
				}

			}
		}
	}

	b.displayedMovesCount += addedCount

	if b.displayedMovesCount >= 9 {

		hasWinner := xb.GetRResult() >= 32600

		if hasWinner {
			number := string((b.displayedMovesCount % 2) + 49) // Number 1 is ascii #49 and 2 is ascii #50.
			dialog.ShowInformation("Player "+number+" has won!", "Congratulations to player "+number+" for winning.", fyne.CurrentApp().Driver().AllWindows()[0])
			b.finished = true
		}
		return
	}
}

func syncPeriodic(b *board, sd *StatusController) {
	ticker := time.NewTicker(time.Millisecond * 400)
	defer ticker.Stop()
	for range ticker.C {
		fyne.Do(func() {
			sync(b, sd)
		})
	}
}

func (i *boardIcon) SetIcon(res fyne.Resource) {
	i.icon.Resource = res
	i.Refresh()
}

func (i *boardIcon) ResetIcon() {
	i.icon.Resource = nil
	i.Refresh()
}