import { Component, inject, input, OnDestroy, output, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar } from '@angular/material/snack-bar';
import { StylePreset } from '../../presets';

// Mirrors the backend's limits, so bad files are rejected before they're uploaded
const ACCEPTED_TYPES = ['image/png', 'image/jpeg'];
const MAX_BYTES = 10 * 1024 * 1024;

@Component({
  imports: [MatButtonModule, MatIconModule],
  selector: 'app-image-picker',
  styleUrl: './image-picker.scss',
  templateUrl: './image-picker.html',
})
export class ImagePicker implements OnDestroy {
  private readonly snackBar = inject(MatSnackBar);

  readonly label = input.required<string>();
  readonly presets = input<StylePreset[]>([]);
  readonly fileSelected = output<File>();

  protected readonly previewUrl = signal<string | null>(null);
  protected readonly selectedPreset = signal<StylePreset | null>(null);
  protected readonly dragging = signal(false);

  protected onFileChange(event: Event) {
    const inputElement = event.target as HTMLInputElement;
    const file = inputElement.files?.[0];
    inputElement.value = ''; // lets the same file be picked again
    if (file) this.select(file);
  }

  protected onDragOver(event: DragEvent) {
    event.preventDefault(); // without this the browser won't allow a drop here
    this.dragging.set(true);
  }

  protected onDrop(event: DragEvent) {
    event.preventDefault(); // stops the browser opening the file in the tab
    this.dragging.set(false);
    const file = event.dataTransfer?.files[0];
    if (file) this.select(file);
  }

  protected async selectPreset(preset: StylePreset) {
    try {
      const response = await fetch(preset.url);
      if (!response.ok) throw new Error(response.statusText);
      const blob = await response.blob();
      const fileName = preset.url.split('/').pop()!;
      this.select(new File([blob], fileName, { type: blob.type }), preset);
    } catch {
      this.snackBar.open(`Couldn't load ${preset.name}. Please try again.`, 'Dismiss', {
        duration: 6000,
      });
    }
  }

  ngOnDestroy() {
    this.revokePreview();
  }

  private select(file: File, preset: StylePreset | null = null) {
    const error = validate(file);
    if (error) {
      this.snackBar.open(error, 'Dismiss', { duration: 6000 });
      return;
    }

    this.revokePreview();
    this.previewUrl.set(URL.createObjectURL(file));
    this.selectedPreset.set(preset);
    this.fileSelected.emit(file);
  }

  // Object URLs keep the file in memory until revoked
  private revokePreview() {
    const url = this.previewUrl();
    if (url) URL.revokeObjectURL(url);
  }
}

function validate(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return 'Only JPEG and PNG images are supported.';
  if (file.size > MAX_BYTES) return 'Images must be 10 MB or smaller.';
  return null;
}
