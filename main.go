//go:generate fyne bundle -o data.go Icon.png

package main

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -ltorch -ltorch_cpu -lc10 -lstdc++
*/
import "C"

import (
    "os"
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/widget"
	"github.com/ratamahata/xgo/xfyneui"
)

type appInfo struct {
	name string
	icon fyne.Resource
	canv bool
	// Signature remains as requested, taking the StatusController
	run func(fyne.Window, *xfyneui.StatusController) fyne.CanvasObject
}

// FIX: Use a wrapper function to pass the StatusController to the UI
var appx = appInfo{
	name: "Tic Tac Toe",
	icon: nil,
	canv: true,
	run: func(w fyne.Window, sc *xfyneui.StatusController) fyne.CanvasObject {
		// If you CANNOT change xfyneui.Show, call it here.
		// If you CAN change xfyneui.Show, update it to accept sc.
		return xfyneui.Show(w, sc)
	},
}

func main() {
    os.Setenv("TORCH_CPP_LOG_LEVEL", "ERROR")

	a := app.New()
	a.SetIcon(resourceIconPng)

	// 1. Initialize Status Data
	statusData := &xfyneui.StatusController{}
	statusContainer := container.NewVBox()
	for i := 0; i < 9; i++ {
		statusData.Lines[i] = binding.NewString()
		statusData.Lines[i].Set("Waiting...")
		statusContainer.Add(widget.NewLabelWithData(statusData.Lines[i]))
	}

	// 2. Create and Show Status Window
	sw := a.NewWindow("Status Monitor")
	sw.SetContent(statusContainer)
	sw.Resize(fyne.NewSize(300, 200))
	sw.Show()

	// 3. Create Main Window
	w := a.NewWindow("X & O")

	// Pass statusData here
	content := container.NewMax(appx.run(w, statusData))

	w.SetContent(content)
	w.Resize(fyne.NewSize(580, 605))
	w.ShowAndRun()
}
