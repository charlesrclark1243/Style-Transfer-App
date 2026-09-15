import { Component, input, signal } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';

@Component({
  imports: [MatIconModule],
  selector: 'app-compare-slider',
  styleUrl: './compare-slider.scss',
  templateUrl: './compare-slider.html',
})
export class CompareSlider {
  readonly before = input.required<string>();
  readonly after = input.required<string>();

  // How far across the image, in percent, the before image is shown
  protected readonly position = signal(50);

  protected onInput(event: Event) {
    this.position.set((event.target as HTMLInputElement).valueAsNumber);
  }
}
