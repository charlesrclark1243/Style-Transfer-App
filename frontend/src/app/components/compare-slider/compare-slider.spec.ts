import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CompareSlider } from './compare-slider';

describe('CompareSlider', () => {
  let component: CompareSlider;
  let fixture: ComponentFixture<CompareSlider>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CompareSlider],
    }).compileComponents();

    fixture = TestBed.createComponent(CompareSlider);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('before', 'before.jpg');
    fixture.componentRef.setInput('after', 'after.jpg');
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should move the divider when the slider changes', async () => {
    const element = fixture.nativeElement as HTMLElement;
    const slider = element.querySelector('input[type="range"]') as HTMLInputElement;

    slider.value = '25';
    slider.dispatchEvent(new Event('input'));
    await fixture.whenStable();

    const compare = element.querySelector('.compare') as HTMLElement;
    expect(compare.style.getPropertyValue('--position')).toBe('25%');
  });
});
