import { ComponentFixture, TestBed } from '@angular/core/testing';
import { STYLE_PRESETS } from '../../presets';
import { ImagePicker } from './image-picker';

describe('ImagePicker', () => {
  let component: ImagePicker;
  let fixture: ComponentFixture<ImagePicker>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImagePicker],
    }).compileComponents();

    fixture = TestBed.createComponent(ImagePicker);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('label', 'Style');
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render a button per preset', async () => {
    fixture.componentRef.setInput('presets', STYLE_PRESETS);
    await fixture.whenStable();

    const buttons = (fixture.nativeElement as HTMLElement).querySelectorAll('.preset');
    expect(buttons.length).toBe(STYLE_PRESETS.length);
  });
});
