import { Component, input, OnDestroy, output, signal } from "@angular/core";
import { MatButtonModule } from "@angular/material/button";

@Component({
  imports: [MatButtonModule],
  selector: "app-image-picker",
  templateUrl: "./image-picker.html",
  styleUrl: "./image-picker.scss"
})
export class ImagePicker implements OnDestroy {
  readonly label = input.required<string>();
  readonly fileSelected = output<File>();

  protected readonly previewUrl = signal<string | null>(null);

  protected onFileChange(event: Event) {
    const inputElement = event.target as HTMLInputElement;
    const file = inputElement.files?.[0] ?? null;
    if (!file) return;

    this.revokePreview();
    this.previewUrl.set(URL.createObjectURL(file));
    this.fileSelected.emit(file);

    inputElement.value = "";
  }

  ngOnDestroy() {
    this.revokePreview();
  }

  // Object URLs keep the file in memory until revoked
  private revokePreview() {
    const url = this.previewUrl();
    if (url) URL.revokeObjectURL(url);
  }
}