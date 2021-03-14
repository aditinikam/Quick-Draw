import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { DatasetComponent } from './dataset/dataset.component';
import { HomeComponent } from './home/home.component';
import { ResultComponent } from './result/result.component';

const routes: Routes = [
  { path: 'result', component: ResultComponent },
  { path: 'dataset', component: DatasetComponent },
  { path: 'home', component:HomeComponent},
  { path: '**', component: HomeComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
