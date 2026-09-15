import { ComponentFixture, TestBed } from '@angular/core/testing';
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
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
